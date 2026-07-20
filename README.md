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

## Measured performance

Dev box (1× RTX 4090 used, 93 GB RAM, NFS storage), Qwen1.5-MoE-A2.7B
single-stream decode, 8 GB GPU expert cache:

| Config | decode tok/s | expert cache hit |
|---|---|---|
| fp16 experts, hybrid CPU/GPU | 6.9 | 44% |
| INT4 (Marlin) GPU experts | **14.6** | 80% |

Dense Qwen3-0.6B fp32: 30 tok/s (python-loop bound; CUDA graphs/paged
attention are the next lever).

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
