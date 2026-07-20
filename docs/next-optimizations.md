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

**TODO: CUDA-graph the decode step.** Capture the T=1 forward once and replay
(~1 launch instead of ~500-800). Requires static addresses + shapes:
- KV cache is already a stable capacity-doubling buffer (`KVCache`); the only
  varying shape is attention over S keys. Options: (a) capture per KV-length
  bucket and re-capture on growth past a power-of-2, or (b) pad K/V to a fixed
  max and mask — pick per measurement.
- Preallocate all decode scratch; no `cp.asnumpy`/host branches inside capture
  (the MoE routing D2H sync must move out — see #2).
Expect the largest single jump here, especially for the INT4 / 67B path.

## 2. Fused MoE kernel (kills per-expert dispatch + routing syncs)

MoE decode does, per layer: a gate matmul + `cp.asnumpy(logits)` (a hard D2H
**sync**, 24×/token) to route on CPU, then a Python loop over top-k experts
each doing cache lookup + 2 H2D copies + 3 Marlin calls + scatter. That
per-expert Python is the MoE floor.

Target: a grouped/batched expert GEMM that takes the routing on-device (top-k
via `cupy` argpartition) and runs all activated experts without a host loop or
per-layer D2H sync. This is what vLLM/sglang fused-MoE kernels do. Large but
it's the only way to get MoE decode competitive.

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
