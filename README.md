# fastllm-py

A Python-native LLM inference engine (CuPy + NumPy + asyncio) that ports
[fastllm](https://github.com/ztxz16/fastllm)'s hybrid CPU/GPU MoE strategy:
run large MoE models on a single consumer GPU + server RAM by executing
hot/bulky experts on the GPU and cold/sparse experts on the CPU,
concurrently.

## What works today

| Capability | Status |
|---|---|
| Config-driven model graph (zero per-model code) | ✅ Qwen3, Qwen2-MoE, DeepSeek-V2 families |
| Dense forward == HuggingFace (fp32, exact argmax) | ✅ Qwen3-0.6B validated |
| Hybrid CPU/GPU MoE (task-count split, concurrent exec) | ✅ Qwen1.5-MoE-A2.7B validated vs HF |
| MLA attention + YaRN RoPE (DeepSeek V2/V3) | ✅ DeepSeek-V2-Lite validated vs official-semantics HF (exact argmax + identical greedy) |
| GPU expert LRU cache (copy-stream uploads, event-ordered) | ✅ |
| Cross-layer expert prefetch (frequency-based) | ✅ |
| Marlin INT4 GEMM + FP8 block-128 GEMV (fastllm CUDA kernels via ctypes) | ✅ .so built, 18 tests |
| FP8-E4M3 block-128 + INT4 group quantizers (numpy/cupy) | ✅ bit-exact vs torch fp8 |
| OpenAI-compatible server (`/v1/chat/completions`, streaming) | ✅ smoke-tested (SSE + JSON, 37 tok/s on Qwen3-0.6B) |
| Marlin INT4 GPU experts (4× cache capacity) | ✅ opt-in `gpu_expert_quant="int4"` |
| Speed-estimator calibration (CPU/GPU crossover threshold) | ✅ |
| FlashInfer paged attention, speculative decoding | ⏳ planned |

## CUDA-graph decode (INT4)

Capturing the whole per-token decode step as a CUDA graph and replaying it as
one launch removes the Python/driver dispatch overhead that dominates decode —
**bit-exact vs eager** (runtime `verify()` + eager fallback), any GPU count
(one graph per device-segment). `fastllm_py/graph_decode.py`.

| Model (INT4) | GPUs | eager | graph | speedup |
|---|---|---|---|---|
| Qwen3-0.6B | 1 | 44 | **228** | 5.2× |
| deepseek-coder-1.3b | 1 | 60 | **277** | 4.6× |
| Qwen3-8B | 1 | 33 | **89** | 2.7× |
| deepseek-llm-67b | 2 | 15.4 | **20.7** | 1.34× |

Speedup shrinks with width (bigger GEMMs are less dispatch-bound) but is a win
everywhere. Attention is a flash-decode RawKernel (O(valid_len), not O(buffer)),
so graph decode is robust at any context length. Detail: `benchmarks/GRAPH_RESULTS.md`.

**On by default.** `Model.generate()` auto-routes greedy decode to a cached
CUDA-graph decoder whenever the model is `graph_capable` (INT4 dense any GPU
count, or resident-INT4 MoE on one GPU, non-MLA), and silently falls back to
eager otherwise (sampling, fp16 weights, MLA, offloaded MoE, or a capture/verify
failure). No flag needed — the low-level API and the server both take the fast
path. Pass `use_graph=False` to force eager.

## Speculative decoding

Greedy speculative decoding (`fastllm_py/speculative.py`) — a small draft
proposes γ tokens, the target verifies in one forward. Output is **identical to
greedy target decoding**. Qwen3-8B target / Qwen3-0.6B draft, both INT4, 1 GPU:
**2.03×** (γ=4, 68% draft acceptance; target forwards 96 → 30). The draft
rollout runs on the graph decoder, so this composes with graph decode.

Unlike graph decode, speculative can't be "on by default" — it needs a second
draft model sharing the target's vocab, which is a model-selection/resource
decision. Opt in at the server with `--draft-model <dir>` (greedy requests
route through it and stream; sampling requests fall back to the normal path):

```bash
.venv/bin/python -m fastllm_py.server --model models/Qwen3-8B \
    --linear-quant int4 --draft-model models/Qwen3-0.6B --spec-gamma 4
```

## Batched decode (throughput / many concurrent streams)

Decode is bottlenecked on reading the weights from VRAM once per token, so
running B sequences together (each with its own KV) amortizes that read across
all B — aggregate throughput scales with batch. `fastllm_py/batched.py`
(`BatchedDecoder`): batched marlin GEMVs, a batched flash-decode attention
kernel (per-sequence length), and CUDA-graph capture of the whole batched step.

Qwen3-0.6B INT4, aggregate tok/s (CUDA-graph, single GPU):

| Batch | aggregate tok/s | vs 1-stream eager |
|---|---|---|
| 1 | 227 | 3.3× |
| 8 | 1056 | 15× |
| 16 | **1399** | **21×** |

Each of the 16 streams still runs ~87 tok/s. Correct: every sequence's output
is identical to its single-stream generation. Dense non-MLA INT4, single GPU.

**Continuous batching** (`BatchedEngine` / `ContinuousEngine`): a fixed pool of
slots; sequences join and leave dynamically (finished slot → next queued
request is prefilled in) without re-capturing the graph. Served via the OpenAI
endpoint with `--continuous --max-batch N` (greedy). Sustained ~700 tok/s
through 16 slots under continuous load (includes per-request prefill).

## Serving

OpenAI-compatible server auto-enables graph decode for INT4 dense models; each
request's whole generation runs on one worker thread and streams via the loop.
Qwen3-0.6B INT4 served at ~190 tok/s.

## MoE — residency, not offload, is the win

For MoE that fits in VRAM at INT4 (Qwen1.5-MoE experts 5.8 GB, V2-Lite,
moe-16b), keep experts **resident** on GPU (`moe_device={"cuda":1}` +
`gpu_expert_quant="int4"`). Then three levels of MoE decode on Qwen1.5-MoE-A2.7B:

| Path | tok/s |
|---|---|
| offloaded (75% experts on CPU) | ~1 |
| resident, eager per-expert (marlin) | ~19 |
| resident, **fused selective kernel** | ~34 |
| resident, **fused kernel + CUDA graph** | **~113** |

The fused kernel (`fastllm_py/kernels/moe_int4.py`) runs only the routed experts
in one launch pair with on-GPU routing; graph-capturing the whole MoE decode
(`GraphDecoder`) then removes the per-token dispatch. Offloaded MoE (the 671B
case) additionally needs prefetch/overlap — see `docs/next-optimizations.md`.

## Scope / what fits this host

2× RTX 4090 (24 GB), 93 GB RAM. Largest model: **deepseek-llm-67b INT4 across
both GPUs** (~18 GB/GPU, 19 tok/s graph decode). 671B-class MoE (DeepSeek V4,
GLM-5.2) does **not** fit even at INT4 (>300 GB weights) — needs a 256 GB+ RAM
box; those also need new DSA sparse-attention code.

## Layout

```
fastllm_py/
  config.py         HF config.json → ModelConfig (dense + MoE + MLA keys)
  weights.py        lazy safetensors WeightStore
  quantizer.py      FP8-E4M3 block-128, INT4 group (numpy/cupy)
  device_router.py  DeviceMap (layers→devices), MoeDeviceMap (experts→tiers)
  model.py          generic decoder: GQA/QK-norm/MLA attention, auto-MoE attach
  expert_router.py  top-k routing (softmax/sigmoid+bias), task split, estimator
  expert_cache.py   GPU LRU cache w/ async uploads (per-stream-arena safe)
  moe.py            concurrent CPU(threadpool)+GPU(stream) expert execution
  engine.py         asyncio generation engine (streaming)
  server.py         OpenAI-compatible FastAPI server
  benchmark.py      split-threshold calibration
  kernels/
    ops.py          RMSNorm/RoPE(+YaRN)/SwiGLU (xp-generic)
    marlin.py       ctypes wrapper: fastllm Marlin INT4 GEMM
    fp8.py          ctypes wrapper: fastllm FP8 block-128 GEMV
native/             CUDA build → libfastllm_kernels.so (bash native/build.sh)
```

## Quick start

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e . dev
bash native/build.sh          # optional: native INT4/FP8 kernels (needs nvcc)

# generate
.venv/bin/python scripts/generate.py models/Qwen3-0.6B "The capital of France is"

# serve (hybrid MoE: 25% experts GPU-resident, rest CPU)
.venv/bin/python -m fastllm_py.server --model models/Qwen1.5-MoE-A2.7B \
    --moe-device '{"cuda": 1, "cpu": 3}' --gpu-cache-gb 8

# benchmark
.venv/bin/python scripts/benchmark_throughput.py models/Qwen1.5-MoE-A2.7B \
    --moe-device '{"cuda": 1, "cpu": 3}' --calibrate
```

## Testing

```bash
.venv/bin/pytest                       # everything available
# heavy end-to-end tests need models/ + reference logits:
.venv/bin/python scripts/make_reference.py models/Qwen1.5-MoE-A2.7B models/qwen15moe_ref.npz
```

Reference: `docs/fastllm-internals.md` documents the C++ internals this port
replicates (expert split heuristics, kernel formats, stream orchestration).

Note: for DeepSeek models we follow the **official** softmax-scale semantics
(`scale *= yarn mscale²`, as in modeling_deepseek.py / vLLM / fastllm).
transformers ≥ 5 native `deepseek_v2` omits this correction, so references
must be generated with `scripts/make_reference_dsv2.py` (which patches it in).
