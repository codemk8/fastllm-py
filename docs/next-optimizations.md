# Next optimizations (decode throughput)

Status after the 2026-07-20 optimization pass: dense decode is ~40-60 tok/s
(1-2B) and the 67B INT4 runs at 12.9 tok/s on 2×4090. The remaining gap to
bandwidth-bound frameworks (vLLM/llama.cpp) is **host-side Python dispatch**,
not GPU work — proven by `scripts/diagnose_moe_decode.py` (MoE decode was
~2 tok/s with 90% cache hits and <20 uploads/token; uploads were ~2ms of a
500ms token). The levers below attack dispatch directly, in priority order.

## 1. Stream-controlled Marlin → CUDA-graph decode capture (biggest lever)

**Problem.** `FastllmCudaMarlinHalfInt4Gemm` hardcodes CUDA **stream 0**
(the legacy default stream) — `fastllm-marlin.cu:2666`, the `dev, 0,` args to
`marlin::marlin_mm<half>`. Stream 0 is an implicit **device-wide barrier**
against blocking streams, so every Marlin GEMM serializes the whole device,
and — critically — **legacy-stream work cannot be captured into a CUDA graph**.

**DONE (2026-07-20): stream-accepting entry.** `FastllmCudaMarlinHalfInt4Gemm
Stream(..., void* stream)` is built into the .so by patching a build-time
*copy* of `fastllm-marlin.cu` (upstream untouched) — see
`native/marlin_stream_entry.inc` + `native/build.sh`. `marlin.gemm_fast(...,
stream=)` routes to it (bit-identical to the default, tested); INT4 MoE
experts now run on `compute_stream` instead of stream 0. This unblocks graph
capture and removes the barrier vs the blocking compute stream.

**DONE: CUDA-graph the decode step** (`fastllm_py/graph_decode.py`). Captures
the whole T=1 INT4-dense step and replays it as one launch. **~4.2-4.9x decode
speedup**, bit-exact vs eager (Qwen3-0.6B 43->213, coder-1.3b 60->277,
R1-1.5B 49->205 tok/s). Design: pad K/V to `max_len` with a bias mask (static
shapes); Marlin linears on the capture stream; attention as broadcast-multiply
+ reductions (cuBLAS is rejected during cupy capture); lm_head runs outside the
graph; new K/V written by a RawKernel at a device-resident position; dedicated
capture mem-pool + per-call Marlin workspaces; all input writes on the capture
stream (a default-stream write race was the long-hunted correctness bug).
Remaining: single-GPU + non-MLA only (67B is 2-GPU; MLA/MoE need the routing
D2H sync moved on-device — see #2).

## 2. Fused MoE kernel (kills per-expert dispatch)

**Measured correction (2026-07-20): the MoE floor was NOT Python dispatch — it
was expert upload/cache-thrash.** With all experts made INT4-**resident** on
GPU (`moe_device={"cuda":1}` + `gpu_expert_quant="int4"` + a cache big enough
to hold them), Qwen1.5-MoE-A2.7B eager decode is **~20 tok/s** — vs ~1 tok/s
when 75% of experts are offloaded to CPU (an 18× gap; upload-bound). So
for MoE that **fits in VRAM at INT4** (Qwen1.5-MoE experts = 5.8 GB, V2-Lite,
moe-16b all fit), the win is simply residency — no kernel work needed. Use the
resident config.

**Profile of resident-MoE decode (Qwen1.5-MoE-A2.7B, 53 ms/tok = 18.8 tok/s;
scripts/profile_moe.py):**
- gate + routing + **D2H sync: 3.1 ms (6%)** — NOT the bottleneck.
- **GPU expert dispatch + GEMV: 31 ms (58%)** — 24L × 4 experts × 3 GEMVs = 288
  GEMV calls/token, **~107 µs each** vs ~4 µs of actual memory-bound work ⇒
  ~96% Python/launch **dispatch**, same disease as dense decode.
- non-MoE (attention/norm/lm_head): 19 ms (36%) — also eager-dispatch-bound.

**Dead-end (reverted), now with the reason:** dense-over-experts CUDA-graph MoE
(compute all E experts, weight-mask to top-k) measured **0.84×**. Decode is
weight-BANDWIDTH-bound, and computing all 60 experts reads **15× more weight
bytes** than the routed 4 — so it can never beat selective, regardless of
dispatch. `route_gpu` (on-GPU group-limited routing) is kept as the routing
half of the real kernel.

**The lever — a fused SELECTIVE MoE kernel** (helps resident decode too, not
only the offload/671B case): read only the routed top-k experts' INT4 weights
(minimal bandwidth) with routing on-device, in one/few launches instead of 288
— removing the dispatch and making the MoE FFN CUDA-graph-capturable (like the
dense path). Design constraint: our experts are quantized in the *Marlin* tiled
layout, which is very hard to gather/dequant in a custom kernel. Plan: a **row-major INT4 group** expert format (`fastllm_py/kernels/moe_int4.py`)
a custom kernel can gather + dequant inline. **Status (2026-07-20):**
- (1) profile — done.
- (2) `gemv_int4` custom row-major INT4 GEMV — done, matches dequant ref.
- (3) `fused_moe_ffn` one-block-per-expert selective FFN — correct but 8× slower
  (only K blocks → GPU idle).
- (4) `fused_moe_ffn2` two-kernel, one block per (expert, output-row), coalesced
  + block-reduce — **5.4× faster than the eager marlin per-expert loop** on the
  routed FFN (48.5 vs 264 µs), one launch pair, reads only routed weights.
- (5) wired into real decode — **Qwen1.5-MoE-A2.7B 18.8 → 33.7 tok/s (1.8×)**,
  tokens match the marlin path. (Decode-only; T>1 prefill falls back to eager.)
- (6) capturable primitives — `gate_matvec` (fp16 gate, no cuBLAS) +
  `fused_moe_weighted` (driven by an (E,) routing-weight vector so no index
  extraction / D2H). Verified gate+route+fused captures as one graph.
- (7) **graph-capture the whole MoE decode** in GraphDecoder (`_moe_branch`):
  attention flash-decode + gate + routing + fused MoE + shared, all inside the
  captured graph. **Qwen1.5-MoE 26 → 112.8 tok/s (4.33× graph-over-eager)**,
  coherent output. Caveat: not bit-exact vs the fp16-shared eager path — the
  shared expert is additionally INT4-quantized (fp16 FFN needs cuBLAS); a small
  quality tradeoff, verify() confirms graph == its own execution.

**Remaining:** custom fp16 shared-FFN kernels for exact parity; batched (B>1)
fused MoE; multi-GPU MoE graph; and the *offload* case (experts don't fit —
671B) still needs prefetch/overlap on top of this. This is a working
ktransformers-style lever end to end (fused kernel + graph capture).

## 3. FlashInfer paged attention + continuous batching

Replace the naive fp32 `_sdpa` + per-sequence KV with FlashInfer varlen/paged
attention. Unlocks (a) faster attention, (b) paged KV (no fragmentation), and
(c) **continuous batching** — assemble a per-step batch in `AsyncEngine`,
run one `Model.forward`, scatter one token per stream. Batch 16-32 is realistic
for MLA/small models (see the compressed-MLA-KV note below); ~150 lines on the
existing asyncio worker once attention is batched.

## 4. Compressed MLA KV cache (DeepSeek)

We currently cache decompressed per-head K/V. MLA's point is the compressed
latent (`kv_lora_rank` + rope, ~576 B/token/layer vs ~100 KB). Caching the
latent and expanding on the fly makes batch-32 KV essentially free for
DeepSeek models and shrinks decode memory traffic. Changes the KV layout, so
fold it into the FlashInfer work.

## Smaller, safe wins already banked (2026-07-20)

- Amortized-O(T) `KVCache` (was O(T²) concatenate).
- Decode causal-mask skip (T=1).
- `marlin.gemm_fast` — no per-call astype/ascontiguousarray/validation.
- Single-token expert indexing (host int + direct `out[idx]+=y`, not
  `cp.add.at`).
- Fused CuPy RMSNorm + SwiGLU (`USE_FUSED_RMSNORM`). Net dense decode +17-30%.
