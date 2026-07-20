# fastllm-py — Handover (moving to a clean RTX 5090 machine)

Last updated 2026-07-20. Read this top-to-bottom before setting up the new box;
the **Blackwell/5090 gotchas (§4)** are the part most likely to bite.

---

## 1. What this is

A Python-native LLM inference engine (CuPy + NumPy + asyncio) that ports
fastllm's hybrid CPU/GPU MoE strategy, with modern decode optimizations. Goal:
run large models on consumer GPUs + server RAM. `fastllm_py/` is ~2 k lines of
Python + a small native CUDA `.so` (Marlin INT4 GEMM + FP8 GEMV borrowed from
fastllm). Config-driven — essentially zero per-model code.

**What works, validated (all bit-exact vs HuggingFace fp32 unless noted):**
- Dense models (Qwen3, Qwen2, Llama/DeepSeek-LLM) — exact vs HF.
- MLA + YaRN (DeepSeek-V2) — exact vs official-semantics reference.
- Hybrid + resident MoE (Qwen2-MoE, DeepSeek-MoE) — exact vs HF.
- Marlin INT4 dense (`linear_quant="int4"`), on-disk `.marlin_cache`.
- **CUDA-graph decode** (INT4 dense, 1..N GPUs) — bit-exact, 2.6–4.9×.
- Speculative decoding (greedy = identical to target).
- DeepSeek V3/V4 group-limited routing.
- OpenAI-compatible server with graph-accelerated decode.

**Honest limitations (don't re-discover these):**
- **671B-class models (DeepSeek V4 671B–1.6T, GLM-5.2 744B) do NOT fit** even
  at INT4 (~336 GB / ~372 GB weights) — needs a 256 GB+ RAM box. This host had
  93 GB. A 5090 box won't change that unless it has huge system RAM.
- **DeepSeek V4 / GLM-5.2 need new architecture code** (DSA sparse attention +
  lightning indexer, MTP) — not implemented. See `docs/next-optimizations.md`.
- **MoE offload (experts don't fit in VRAM) is upload-bound**, and the only fix
  is a fused selective gather-GEMV kernel (multi-week). For MoE that *fits* at
  INT4, just make experts resident (below) — that was the real ~4× win.
- CUDA-graph decode is **dense non-MLA + INT4 only** (cuBLAS/MLA-routing can't
  be captured); MLA/MoE fall back to eager.

---

## 2. Validated performance (on the old 2×RTX 4090 box)

Use these as the **regression baseline** — after porting, numbers should be
*higher* on a 5090 (more bandwidth), and correctness must still be exact.

| Model | Config | Decode tok/s |
|---|---|---|
| Qwen3-0.6B INT4 | 1 GPU, **graph** | 213 (eager 43) |
| deepseek-coder-1.3b INT4 | 1 GPU, graph | 277 (eager 60) |
| Qwen3-8B INT4 | 1 GPU, graph | 89 (eager 34) |
| deepseek-llm-67b INT4 | 2 GPU dense | 12.9 |
| Qwen1.5-MoE-A2.7B INT4 | experts **resident** | **~26** (eager) |
| Qwen1.5-MoE-A2.7B INT4 | 25% experts on CPU | 6.4 (upload-bound) |

Server: Qwen3-0.6B INT4 served at ~190 tok/s (graph decode in the async engine).

Two clean measurements were still **pending** when we handed off (co-tenant GPU
contention on the old box): the **67B 2-GPU graph** decode (`scripts/run_67b_graph.py`
— eager was 14.7; graph-vs-eager verdict unconfirmed after a pipeline fix) and a
clean **speculative** run (`scripts/test_speculative.py` — ~1.3× under
contention, correctness confirmed). Re-run these first on the clean box.

---

## 3. Setup on the new machine (step-by-step)

```bash
# 0. clone (the repo has the code; models are separate, see §5)
git clone <your-remote>/fastllm-py && cd fastllm-py
#    also clone fastllm source (needed only to BUILD the native .so):
git clone --depth 1 https://github.com/ztxz16/fastllm

# 1. venv (uv recommended). Python 3.12.
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e . dev
#    core deps: cupy-cuda12x, numpy, safetensors, transformers>=5, tokenizers,
#    huggingface_hub, hf_transfer, sentencepiece, torch, accelerate, pytest
#    serving:   fastapi, uvicorn
#    NOTE: sentencepiece is REQUIRED for Llama/DeepSeek-LLM tokenizers.

# 2. build the native kernels — SET THE ARCH (see §4!)
ARCH=sm_120 NVCC=/usr/local/cuda-XX.Y/bin/nvcc bash native/build.sh

# 3. sanity-check the stack
.venv/bin/python -c "import cupy; cupy.ones(1000).sum(); print('cupy OK', cupy.__version__)"
.venv/bin/pytest tests/test_native_kernels.py tests/test_quantizer.py -q   # kernels + quant
.venv/bin/pytest tests/test_router.py tests/test_kvcache.py -q             # CPU-only logic

# 4. get a model + run
#    (download to LOCAL disk, not a network mount — see §6 lesson)
uvx --from huggingface_hub hf download Qwen/Qwen3-8B --local-dir models/Qwen3-8B --exclude "*.pth"
.venv/bin/python scripts/test_graph_decode.py models/Qwen3-8B   # validates graph==eager + speed
```

---

## 4. ⚠️ Blackwell / RTX 5090 specifics (do these or nothing runs)

The 5090 is **Blackwell, compute capability `sm_120`** (consumer Blackwell), not
Ada (`sm_89`). Several things are arch/version sensitive:

1. **Rebuild the native `.so` for `sm_120`.** `native/build.sh` defaults to
   `ARCH=sm_89`. Run `ARCH=sm_120 bash native/build.sh`. If nvcc rejects
   `sm_120`, your CUDA toolkit is too old (need **CUDA 12.8+** for Blackwell;
   12.9 or 13.x preferred). The Marlin kernel is vLLM-derived and needs sm_75+;
   it *should* compile for sm_120, but **verify `tests/test_native_kernels.py`
   passes** — if Marlin's tuned configs misbehave on Blackwell, that test
   catches it (compare rel-Frobenius error). FP8 kernels similarly.
2. **CuPy must match the CUDA runtime.** `cupy-cuda12x` works on Blackwell with
   CUDA 12.8+. Confirm `cupy.ones(1000).sum()` runs. If you use CUDA 13, install
   the matching cupy wheel.
3. **PyTorch** is only used for weight loading (bf16→fp32) — install a build with
   Blackwell support (torch 2.7+ / cu128 or cu130). It's CPU-path only here, so
   even a CPU torch works for loading, but a CUDA torch is fine too.
4. **The CUDA-graph decode has a known hard requirement**: input writes must be
   issued on the capture stream (a default-stream race was a nasty bug we fixed
   — see §7). This is arch-independent, but **re-run `scripts/test_graph_decode.py`
   on the 5090** to confirm graph==eager bit-exact — capture semantics can shift
   subtly across driver/cupy versions, and the `verify()` gate will fall back to
   eager if they diverge (so you'll see `[eager-fallback]` instead of `[graph]`).
5. **32 GB VRAM (vs 24)** changes what fits: a single 5090 holds ~14B-class INT4
   dense with graph decode; the 67B INT4 (~36 GB) still needs 2 GPUs but is
   comfy on 2×5090 (64 GB). Bump `graph_max_len` / cache sizes accordingly.
6. **Expected perf**: decode is memory-bandwidth-bound in the limit; the 5090's
   ~1.8 TB/s (vs 4090 ~1 TB/s) should raise absolute tok/s. Graph decode already
   removed dispatch overhead, so its *relative* speedup may shrink but absolute
   goes up. Re-benchmark; don't assume the old ratios.

---

## 5. Code map

```
fastllm_py/
  config.py        HF config.json -> ModelConfig (dense/MoE/MLA/V3-group keys)
  weights.py       lazy safetensors loader (bf16 via torch)
  quantizer.py     FP8-E4M3 block-128 + INT4 group (numpy/cupy), bit-exact vs torch
  device_router.py DeviceMap (layers->GPUs), MoeDeviceMap (experts->cuda/cpu/disk)
  model.py         generic decoder: GQA/QK-norm/MLA attention, KVCache (amortized-O(T),
                   .truncate for spec), linear_quant=int4 marlin path + .marlin_cache,
                   matmul_w dispatch, to_device
  expert_router.py route_topk: softmax + sigmoid+bias + V3/V4 GROUP-LIMITED routing
  expert_cache.py  GPU expert LRU (per-stream-arena-safe uploads, deferred eviction)
  moe.py           hybrid CPU+GPU experts; gemm_fast marlin path; build_marlin_expert_payload
  graph_decode.py  GraphDecoder: per-device-segment CUDA-graph decode (INT4 dense),
                   route_gpu (capturable on-GPU routing helper, for a future fused MoE)
  speculative.py   SpeculativeDecoder (greedy = identical to target); optional graph draft
  engine.py        AsyncEngine: FIFO, auto graph-decode, one-thread-per-request streaming
  server.py        OpenAI /v1/chat/completions (stream + non-stream); --linear-quant int4
  kernels/
    ops.py         RMSNorm/RoPE(+YaRN,+linear)/SwiGLU; fused CuPy variants (USE_FUSED_RMSNORM)
    marlin.py      ctypes wrapper: FastllmCudaMarlinHalfInt4Gemm(+Stream), gemm_fast, repack
    fp8.py         ctypes wrapper: FP8 block-128 GEMV
native/            build.sh -> libfastllm_kernels.so; marlin_stream_entry.inc (build-time patch)
docs/              fastllm-internals.md (the C++ we ported), next-optimizations.md (roadmap), this file
scripts/           test_graph_decode.py, test_graph_multigpu.py, run_67b_graph.py,
                   test_speculative.py, bench_suite.py/bench_one.py, make_reference*.py
tests/             pytest suite (see §8)
```

---

## 6. Lessons / gotchas that cost real debugging time

- **CuPy per-stream memory pools**: allocate a buffer under the stream that will
  *write* it, or a queued kernel on another stream corrupts it (silent NaN).
  Record an event + `wait_event` across streams. (`expert_cache.py`.)
- **CUDA-graph capture**: cuBLAS is rejected during capture → INT4/Marlin only,
  attention as reductions, lm_head outside the graph. Intermediates need a
  **dedicated mem-pool** and **per-call Marlin workspaces** (a shared workspace
  corrupts a replayed graph). The killer bug was a **cross-stream input-write
  race** (found via `compute-sanitizer`: 0 mem errors + correct output ⇒ race,
  not corruption) — issue input writes on the capture stream.
- **transformers 5**: `apply_chat_template(tokenize=True)` returns a
  BatchEncoding; native `deepseek_v2` omits the YaRN mscale² softmax correction
  that official/vLLM/fastllm apply — generate references with
  `scripts/make_reference_dsv2.py`.
- **MoE floor was upload/cache-thrash, NOT Python dispatch.** Make experts
  INT4-resident (`moe_device={"cuda":1}` + `gpu_expert_quant="int4"` + big cache)
  → ~4×. Dense-over-experts-in-a-graph was a measured dead-end (0.84×).
- **Local disk, not NFS.** The old box's `/data` NFS mount stalled constantly
  (D-state hangs on venv/model reads). Keep venv + models on local disk. On the
  new box this hopefully moots itself, but don't put the venv on a network mount.
- **numpy fp32@fp16 matmul** silently skips BLAS (~10× slower) — cast to fp32.

---

## 7. Test suite / how to validate the port

```bash
.venv/bin/pytest tests/ -q          # full suite
```
Groups: `test_quantizer` (FP8 bit-exact vs torch), `test_native_kernels`
(Marlin/FP8 .so — **the sm_120 canary**), `test_router` (V3 group routing),
`test_kvcache`, `test_fused_ops` (fused RMSNorm/SwiGLU), `test_moe` (synthetic
MoE incl. marlin), `test_dense_int4`. Model-level tests (`test_forward`,
`test_deepseek`, `test_moe_model`, `test_model_zoo`) need models + saved
references under `models/` and `models/refs/` (regenerate with
`scripts/make_reference*.py`). **First thing to run on the 5090: the native
kernel tests** — they confirm the recompiled `.so` is correct for Blackwell.

---

## 8. Data migration (models, caches)

- **Models** lived at `/opt/tmp/ypzhang/models/` on the old box (local disk):
  Qwen3-0.6B, Qwen3-8B, deepseek-coder-1.3b, R1-Distill-1.5B, deepseek-llm-7b,
  deepseek-llm-67b, Qwen1.5-MoE-A2.7B, DeepSeek-V2-Lite, deepseek-moe-16b, plus
  `refs/` (npz references) and tiny test models. Re-download from HF on the new
  box (simplest) or copy over a fast link.
- **`.marlin_cache/`** (INT4 quantized payloads, one `.npz` per matrix) lives
  inside each model dir. The Marlin repack layout is arch-independent, so caches
  are *probably* portable — but the safe path is to **let them regenerate on
  first INT4 load** (a one-time cost: seconds for small models, ~35–54 min for
  the 67B). If you copy them and see garbage, delete `.marlin_cache/` to force a
  rebuild. Cache key is the weight *name* only (no content hash) — delete the
  dir if you ever swap weights at the same path.

---

## 9. Open work / prioritized roadmap (see docs/next-optimizations.md)

1. **Clean re-benchmark on the 5090** — the 67B 2-GPU graph verdict and
   speculative speedup were never measured contention-free. Do these first;
   update `benchmarks/RESULTS.md` + `GRAPH_RESULTS.md`.
2. **Capture the resident-MoE win** in the benchmark table (~26 vs 6.4 tok/s for
   Qwen1.5-MoE) — currently the table understates MoE by using an offloaded config.
3. **Fused selective MoE gather-GEMV kernel** — the only lever for *offloaded*
   MoE (the 671B case). `route_gpu` is the validated routing half; needs the
   gather+GEMV kernel. Multi-week, real CUDA work.
4. **FlashInfer paged attention + continuous batching** — enables multi-request
   throughput (the subagent/batch-16-32 use case). Watch the torch/cupy/CUDA
   version interplay on Blackwell.
5. **DeepSeek V4 / GLM-5.2 architecture** (DSA sparse attention + lightning
   indexer, MTP) — only worth it on a 256 GB+ RAM box where the weights fit.
6. **Multi-GPU MoE + MLA graph decode**, wire speculative into the server,
   prefix/prompt caching.

---

## 10. Quick reference — how to run

```bash
# generate (single GPU, graph decode auto-on for int4)
.venv/bin/python scripts/test_graph_decode.py models/Qwen3-8B

# 67B across 2 GPUs, eager vs graph
.venv/bin/python scripts/run_67b_graph.py models/deepseek-llm-67b-chat

# OpenAI server (graph decode auto-enabled for int4 dense)
.venv/bin/python -m fastllm_py.server --model models/Qwen3-8B \
    --linear-quant int4 --port 8000
#   2-GPU 67B: add --device '{"cuda:0":1,"cuda:1":1}'

# MoE — make experts RESIDENT (the real win) for models that fit at int4:
#   Model.load(path, DeviceMap({"cuda:0":1}), linear_quant="int4",
#              moe_device=MoeDeviceMap({"cuda":1}), gpu_expert_quant="int4",
#              gpu_cache_bytes=12<<30)

# speculative decoding (target + small draft, same vocab)
.venv/bin/python scripts/test_speculative.py models/Qwen3-8B models/Qwen3-0.6B
```

Commit `171510c` is the handoff HEAD. Everything above is on `main`, local commits
only (nothing pushed to a remote yet — **push before wiping the old box**).
