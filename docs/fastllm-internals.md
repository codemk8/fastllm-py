# fastllm Internals: Hybrid MoE Scheduling and Kernel Reference

Source root for all paths below: `/data/user/ypzhang/dev/github/codemk8/fastllm-py/fastllm`
(upstream: https://github.com/ztxz16/fastllm). Line numbers are as of the checked-out
commit (`e801e12`, "first commit") and will drift with upstream changes — function/class
names are the stable anchors.

This document is written as an implementation reference for a Python/ctypes port. It
favors exact signatures and struct layouts over prose.

---

## 1. Hybrid CPU/GPU MoE forward loop

The op is registered as `"MergeMOE"` by both `NumasDevice` (`src/devices/numas/numasdevice.cpp:167`)
and `CudaDevice` (`src/devices/cuda/cudadevice.cpp:2504`). The interesting hybrid logic lives in
the NUMA device's `NumasMergeMOE::Run`, because that is the op instance actually selected when a
layer's MoE weights are declared on `numa` device in the device map — it is the one that decides,
per forward call, which experts run on CPU (NUMA-aware AVX/NEON GEMM) vs which run on GPU
(a background `std::thread` calling into the CUDA device's kernels).

### 1.0 Entry point and batch-size fast path

`NumasMergeMOE::Run` (`src/devices/numas/numasdevice.cpp:2057`):

```cpp
void NumasMergeMOE::Run(const std::string &opType, const fastllm::DataDict &datas,
                const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams)
```

Inputs pulled out of `datas`/`floatParams`/`intParams` (dict keys, `numasdevice.cpp:2063-2084`):
- `input` `[n, hidden]`, `index` `[n, topk]` (int32 expert ids), `score` `[n, topk]` (float32 gate
  weights), `w1/w2/w3` (CUDA-side stacked expert weight tensors, used only for the GPU path),
  `weights`/`biass` (raw `Data**` arrays, length `weightsBatch = (topk+1)*2`; slot `0/1` is the
  shared expert's gate/down, slots `2e/2e+1` are routed expert `e`'s gate/down), `sharedScale`
  (float, default 1.0 — the fixed weight applied to the shared expert's output), `layer` (int,
  used only to pick a double-buffered scratch-space slot, see 1.4).

Two structurally different code paths exist, selected on batch size (`numasdevice.cpp:2147`):

- **`input.dims[0] < 32`** (single-token/small decode batches): a NUMA work-stealing CPU-only
  fast path (`numasdevice.cpp:2147-2481`). It never considers the GPU at all — no `expertLimit`,
  no `gpuPrefill`. This matters for the port: the CPU/GPU split logic below only ever activates
  for batch ≥ 32 (i.e. prefill, or large speculative-decode batches).
- **`input.dims[0] >= 32`**: the CPU/GPU hybrid path described in the rest of this section
  (`numasdevice.cpp:2482-2613`).

A third, homogeneity check runs first regardless of batch size (`numasdevice.cpp:2118-2145`):
if the active experts require more than one CPU activation-quantization format (e.g. a Q8_0
shared expert mixed with Q4_K/Q5_K routed experts in a GGUF-quantized model), it forces the
plain `DoNumasMergeMOEOnCPU` path over *all* active experts and returns — no GPU split in that
case either.

### 1.1 `expertLimit` split threshold

Config singleton `MoeEnvConfig` (`numasdevice.cpp:87-147`), lazily constructed from environment
variables:

| Env var | Effect | Default |
|---|---|---|
| `FT_EXPERT_LIMIT` | Hard override for `expertLimit`; when set, the dynamic benchmark (1.2) is skipped entirely | unset → dynamic |
| `FT_GPU_PREFILL` | `0`/`false`/`off` disables GPU offload for MoE entirely | on |
| `FT_PINNED_WEIGHT` | `0`/`false`/`off` disables pinned-memory allocation for NUMA-sharded expert weights | on |

`expertLimit` starts at `128` (`numasdevice.cpp:101`) and is a **task-count** threshold, not an
expert-id threshold: each expert accumulates a task list of `(token_row, gate_score)` pairs
across the batch (`expertTasks[e]`, built at `numasdevice.cpp:2509-2518` — expert `0` is the
shared expert and always gets one task per row with weight `sharedScale`; routed expert
`indexData[b*topk+j]+1` gets a task per selected row with weight `scoreData[b*topk+j]`). An
expert is then assigned to GPU if its **task count** ≥ `expertLimit`, else CPU
(`numasdevice.cpp:2532-2543`):

```cpp
std::unordered_set<int> cpuExperts, gpuExperts;
for (int e = 0; e < (int)expertTasks.size(); e++) {
    if (weights[e * 2] == nullptr) continue;
    if ((int)expertTasks[e].size() < expertLimit) cpuExperts.insert(e);
    else gpuExperts.insert(e);
}
```

Intuition: an expert selected by many tokens in the batch (large row-count, e.g. the shared
expert, or a "hot" expert in a skewed batch) is more efficient on GPU (its GEMM becomes
compute-bound at larger M); an expert selected by only a few tokens is cheaper to run on CPU
(GPU dispatch/copy overhead dominates for a 1-8 row matmul). If `gpuPrefill` is false (env
override, no CUDA build, or `input.cudaData == nullptr`, or `CanUseCudaMoePrefill` rejects the
activation/weight dtype combo — `numasdevice.cpp:2490-2502`), `expertLimit` is forced to
`INT_MAX` so every expert goes to `cpuExperts` (`numasdevice.cpp:2504-2506`).

`CanUseCudaMoePrefill` (`numasdevice.cpp:786-809`) checks, for every non-null weight slot, that
`IsCudaLinearDataTypeSupported(inputType, weightType, FLOAT32)` holds for the CUDA linear
dispatch table — i.e. the GPU offload path is only enabled when the CUDA device actually has a
kernel for that (activation dtype, weight quant dtype) pair.

### 1.2 Dynamic threshold benchmark — `MoeExpertSpeedEstimator`

Class at `numasdevice.cpp:1028-1344`, singleton via `GetInstance()`. Invoked from the main Run
path only when `gpuPrefill && !hasExpertLimitOverride` (`numasdevice.cpp:2522-2530`):

```cpp
int GetDynamicExpertLimit(
    Data &input, Data &output, Data &w1, Data &w2, Data &w3,
    Data **weights, Data **biass, int weightsBatch, int topk, float sharedScale,
    const std::vector<std::vector<std::pair<int, float>>> &expertTasks,
    int defaultExpertLimit)
```

Mechanism:
1. Build a `MoeBenchmarkShapeKey{inputDim, interDim, outputDim, topk, inputType, outputType,
   gpuId}` (`numasdevice.cpp:995-1026`) and look up/build a per-shape `BenchmarkProfile`
   (`cpuTimeUs`/`gpuTimeUs`: `std::map<int rowCount, double microseconds>`), cached forever in
   `profiles` (`std::unordered_map<MoeBenchmarkShapeKey, BenchmarkProfile>`, guarded by
   `profileLocker`). Rebuilt only if uninitialized or if a larger `adaptiveN` is now needed
   (`profile.maxN < adaptiveN`).
2. `BuildProfile` (`numasdevice.cpp:1226-1343`) picks **one representative routed expert**
   (`PickBenchmarkExpert`, first non-null expert with index ≥ 1) and benchmarks it in isolation
   at a geometrically-spaced set of row counts (`GenerateSamplePoints`/`GetStepForN`,
   `numasdevice.cpp:1160-1188`: step 1 up to N=16, then 4/16/64/128 growing with N, CPU steps are
   `2x` the GPU steps since CPU curves are smoother) up to
   `adaptiveN = min(maxTaskSize, hardMaxBenchmarkN=512, input.dims[0])`. For each sample point it
   runs `warmupRounds=2` warmup calls then `measureRounds=8` timed calls of
   `DoNumasMergeMOEOnCPU(...)` (wall clock, `std::chrono::steady_clock`) and, under `#ifdef
   USE_CUDA`, of `DoCudaMergeMOEFromCPU(..., sharedScale, /*setZero=*/true, gpuExperts,
   /*isCrossSwiglu=*/true)` followed by `FastllmCudaStreamSynchronize(nullptr)` to force the
   async GPU work to complete before stopping the clock.
3. `GetDynamicExpertLimit` then, for every candidate threshold `t` in `[1, maxTaskSize+1]`
   (`numasdevice.cpp:1103-1123`), sums `InterpolateFromMap(cpuTimeUs, size)` for every expert
   whose task count `< t` and `InterpolateFromMap(gpuTimeUs, size)` for every expert whose task
   count `>= t`, and picks the `t` that **minimizes `|cpuTime - gpuTime|`** — i.e. it solves for
   the split that best balances predicted CPU wall time against predicted GPU wall time (since
   the two run concurrently, the effective forward time is `max(cpuTime, gpuTime)`, and
   balancing minimizes that max). `InterpolateFromMap` (`numasdevice.cpp:1190-1215`) does linear
   interpolation between the two bracketing sampled row-counts, or linear extrapolation past the
   last sample.
4. Result is `min(defaultExpertLimit, bestLimit)` (`numasdevice.cpp:2523`) — the benchmark can
   only tighten (shift more experts to CPU relative to) whatever `expertLimit` already is, never
   loosen it.

Python-port note: this whole class can be replaced by an offline-fit or a simpler online model,
but the *shape key* (all of hidden/inter/output dim, topk, in/out dtype, gpu id) and the
*task-count-indexed* CPU/GPU time curves are the right state to reproduce, and the balance
criterion (`argmin |cpu(t) - gpu(t)|` over threshold `t`) is the right optimization target given
CPU and GPU experts run concurrently.

### 1.3 Concurrent CPU+GPU execution and merge

All in `NumasMergeMOE::Run`'s `else` branch, `numasdevice.cpp:2546-2612`:

- **All-GPU fast path**: if `gpuPrefill && cpuExperts.empty()`, just call
  `DoCudaMergeMOEFromCPU(...)` synchronously and return, tagging `output.dataDevice = CUDA`
  (`numasdevice.cpp:2549-2557`).
- **Hybrid path** (`numasdevice.cpp:2559-2609`):
  1. If `!gpuExperts.empty()`, spawn a `std::thread gpuThread` that calls
     `FastllmCudaSetDevice(gpuId)` then `DoCudaMergeMOEFromCPU(input, output, index, score, w1,
     w2, w3, weights, biass, sharedScale, /*setZero=*/true, gpuExperts, /*isCrossSwiglu=*/true)`
     (`numasdevice.cpp:2564-2569`). This thread runs **concurrently** with the CPU work below —
     it is launched before the CPU call, not awaited yet.
  2. Concurrently (main thread), if `!cpuExperts.empty()`: obtain a **pinned host output
     buffer** via `fastllmMoeDataManagerNumas.EnsurePinnedOutput(output.GetBytes())` (only if
     GPU is also active — pure-CPU calls write directly into `output.cpuData`), then run
     `DoNumasMergeMOEOnCPU(input, output, index, score, weights, biass, sharedScale,
     weightsBatch, topk, cpuExperts, fastllmMoeDataManagerNumas, cpuOutputPinned)`
     (`numasdevice.cpp:2573-2582`) which writes the CPU partial MoE output straight into that
     pinned buffer (see 1.4 for `DoNumasMergeMOEOnCPU`'s `cpuOutputBuffer` parameter).
  3. Immediately after the CPU call returns, **while `gpuThread` may still be running**, the
     main thread kicks off an async host→device copy of the pinned CPU partial result onto a
     reusable GPU staging buffer, on a dedicated copy stream
     (`numasdevice.cpp:2584-2593`):
     ```cpp
     cpuOutputStaging = fastllmMoeDataManagerNumas.EnsureGpuOutputStaging(outputBytes, gpuId);
     cpuOutputCopyStream = fastllmMoeDataManagerNumas.EnsureGpuOutputCopyStream(gpuId);
     FastllmCudaCopyFromPinnedHostToDeviceAsync(cpuOutputStaging, cpuOutputPinned, outputBytes, cpuOutputCopyStream);
     ```
  4. Then `gpuThread.join()` (`numasdevice.cpp:2599`) — this is the actual synchronization point
     between CPU-expert compute and GPU-expert compute.
  5. If a CPU partial exists (`cpuOutputStaging != nullptr`), synchronize the copy stream, then
     do an on-device element-wise add of the (already-DMA'd) CPU partial into the GPU partial
     that `DoCudaMergeMOEFromCPU` wrote directly into `output.cudaData`
     (`numasdevice.cpp:2601-2606`):
     ```cpp
     FastllmCudaStreamSynchronize(cpuOutputCopyStream);
     Data gpuOutputAlias(output.dataType, output.dims, DataDevice::CUDA, output.cudaData);
     Data cpuOutputAlias(output.dataType, output.dims, DataDevice::CUDA, cpuOutputStaging);
     FastllmCudaAddTo(gpuOutputAlias, cpuOutputAlias, 1.0f);
     ```
     `FastllmCudaAddTo` (`include/devices/cuda/fastllm-cuda.cuh:322`:
     `bool FastllmCudaAddTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha)`)
     computes `input0 += alpha * input1` — here `alpha=1.0`, a plain sum, because the CPU/GPU
     split already partitions experts disjointly (no double counting) and both partials were
     computed with `setZero=true`/zero-initialized outputs restricted to their own expert set.
  6. `output.dataDevice = CUDA; output.dataDeviceIds = {gpuId};` — final result always ends up
     tagged as living on GPU whenever any GPU expert ran.

Score weighting is folded in per-partial, not at merge time: both `DoNumasMergeMOEOnCPU` and
`DoCudaMergeMOEFromCPU` multiply each expert's down-projection output by its
`(token_row, gate_score)` weight during their own reduce/scatter step (see 1.4/1.5) — by the
time the two partials reach the final `AddTo`, they are both already score-weighted sums over
disjoint expert subsets, so merging is a pure add.

### 1.4 Pinned/staging buffer manager — `FastllmMoeDataManagerNumas`

`numasdevice.cpp:502-562`. One instance per (layer parity) via
`fastllmMoeDataManagerNumasPerLayer[layer % 2]` (`numasdevice.cpp:564, 2084`) — double-buffered
across layers so that layer `L`'s async GPU copy/compute can't race with layer `L+1`'s reuse of
the same scratch buffers (2 slots is enough because only one layer's hybrid MoE is ever
in flight at a time per NUMA-device call site, but the *previous* layer's async copy may still
be draining).

```cpp
struct FastllmMoeDataManagerNumas {
    std::vector<float, alignedAllocator<float,64>> gateUpOutput, swigluOutput, downOutput, reduceOutput;
    std::vector<float, alignedAllocator<float,64>> inputFloat32;
    std::vector<uint8_t, alignedAllocator<uint8_t,64>> realInput, expandInput, downInput;
#ifdef USE_CUDA
    std::unique_ptr<void, FastllmCudaHostFreeDeleter> pinnedOutput;      // cudaHostAlloc'd
    size_t pinnedOutputBytes;
    std::unique_ptr<void, FastllmCudaFreeDeleter> gpuOutputStaging;      // cudaMalloc'd
    size_t gpuOutputStagingBytes; int gpuOutputStagingDevice;
    std::unique_ptr<void, FastllmCudaStreamDestroyDeleter> gpuOutputCopyStream;
#endif
    uint8_t *EnsurePinnedOutput(size_t bytes);              // numasdevice.cpp:516
    void *EnsureGpuOutputStaging(size_t bytes, int gpuId);  // numasdevice.cpp:524
    void *EnsureGpuOutputCopyStream(int gpuId);             // numasdevice.cpp:544
};
```

All three `Ensure*` methods are grow-only caches (realloc only if the requested size exceeds
what's already allocated, or — for the staging buffer/stream — if the target GPU id changed).
`EnsurePinnedOutput` calls `FastllmCudaHostMalloc(bytes)` (page-locked host memory via
`cudaHostAlloc`-style allocator, `include/devices/cuda/fastllm-cuda.cuh:258`).
`EnsureGpuOutputCopyStream` creates a non-blocking stream via `FastllmCudaStreamCreate(true)`
(`fastllm-cuda.cuh:105`).

### 1.5 GPU expert loop — `DoCudaMergeMOEFromCPU`

`src/devices/cuda/cudadevice.cpp:5061-5342`:

```cpp
void DoCudaMergeMOEFromCPU(Data &input, Data &output, Data &index, Data &score,
    Data &w1, Data &w2, Data &w3, Data **weights, Data **biass, float sharedScale,
    bool setZero, const std::unordered_set<int> &experts, bool isCrossSwiglu,
    MoeGateType gateType = MoeGateSwiglu);
```
(forward-declared identically at `numasdevice.cpp:980-982`.)

Flow:
1. Ensures `output`/`input` live on the current CUDA device (`ToCudaTemporary`), zeroing
   `output.cudaData` first if `setZero` (this is why CPU and GPU partials can just be summed —
   each is zero outside its own expert set).
2. Rebuilds `expertTasks[e] = [(token_row, gate_weight), ...]` from `index`/`score` exactly like
   the CPU side (shared expert = index `0` with weight `sharedScale`), flattens them into
   `indexVec`/`scales` + a `startIdx` offset table, and uploads those to device
   (`cudaIndex`, `cudaScales` via `FastllmCudaMalloc`+`FastllmCudaCopyFromHostToDevice`).
3. Allocates reusable `static Data tempInput, tempMiddle, tempSwiglu, tempOutput` scratch tensors
   sized for the batch.
4. **Software-pipelined expert loop** over only the experts in `experts` (`isValidExpert`/
   `findNextValidExpert` skip zero-task or absent experts) using one dedicated `copyStream`
   (`FastllmCudaStreamCreate(true)`) and one `computeDoneEvent`
   (`FastllmCudaEventCreate()`), `cudadevice.cpp:5184-5327`:
   - Before the loop, prefetch expert 0's gate/down weights to device synchronously
     (`weights[curExpert*2]->ToCudaTemporary({}, true)`), no stream arg → default stream.
   - Each iteration: (a) if a next expert exists, kick off its weight upload **asynchronously on
     `copyStream`** (`weights[nextExpert*2]->ToCudaTemporary({}, true, copyStream)`) — this is
     the "prefetch next expert's weights while computing this one" double-buffer; (b) gather this
     expert's assigned rows out of `input.cudaData` via `FastllmCudaPickInput` (`fastllm-cuda.cuh:97`:
     `void FastllmCudaPickInput(uint8_t *input, uint8_t *partInput, int rows, int cols, int
     *cudaIndex)` — a scatter/gather-by-index kernel); (c) run gate GEMM
     (`DoCudaLinearReshape`+`DoCudaLinear`) → MoE activation (`ApplyCudaMoeGate`, SwiGLU/GeGLU
     per `gateType`, with `isCrossSwiglu` controlling whether gate/up are interleaved or
     concatenated) → down GEMM; (d) scatter-add the per-expert result back into `output.cudaData`
     at the original token rows, **scaled by that expert's gate score**, via
     `FastllmCudaPickOutput` (`fastllm-cuda.cuh:98`: `void FastllmCudaPickOutput(uint8_t
     *partOutput, uint8_t *output, int rows, int cols, int *index, float *scales, DataType
     dataType)` — this is where GPU-side score weighting happens, symmetric to the CPU reduce
     step in 1.4/1.6); (e) `FastllmCudaEventRecord(computeDoneEvent)` on the default stream, then
     `FastllmCudaStreamWaitEvent(copyStream, computeDoneEvent)` so the *next* weight-eviction
     (freeing the *previous* expert's device buffer) doesn't race the compute that's still
     reading it; if a next expert exists, `FastllmCudaStreamSynchronize(copyStream)` before
     moving on (so its weights are fully resident before that iteration computes); free the
     *previous* expert's temporary CUDA weight buffers (`FreeCudaTemporary`) only after the
     event fires.
   - After the loop, `FastllmCudaEventSynchronize` + free the last expert's buffers, destroy the
     event/stream, free `cudaIndex`/`cudaScales`.

This is the mechanism referenced by item 6 of the task: GPU experts run on a private stream
concurrent with the (separate, `std::thread`-hosted) CPU experts; within the GPU thread itself,
weight *prefetch* is further pipelined one expert ahead of *compute* using a second private
stream + event, all while the CPU thread and its own memcpy pipeline run independently on the
host. Note that the `output.cudaData` accumulation here is *not* toward a pinned buffer — pinning
is only used for the *CPU→GPU* partial-result handoff (1.3/1.4); the GPU-internal weight
streaming uses ordinary (non-pinned-host, purely device-resident) buffers because weights are
already CUDA-resident `Data` objects being paged in/out of the device, not host memory being
DMA'd in.

### 1.6 CPU expert loop — `DoNumasMergeMOEOnCPU`

`numasdevice.cpp:984-993` (decl) / `numasdevice.cpp:1346-1801` (body):

```cpp
void DoNumasMergeMOEOnCPU(
    Data &input, Data &output, Data &index, Data &score,
    Data **weights, Data **biass, float sharedScale,
    int weightsBatch, int topk, const std::unordered_set<int> &cpuExperts,
    FastllmMoeDataManagerNumas &fastllmMoeDataManagerNumas,
    uint8_t *cpuOutputBuffer   // non-null => write partial result here instead of output.cpuData
);
```

Steps (mirrors the GPU loop's gather/compute/scatter shape but on CPU, NUMA-sharded):
1. Rebuilds `expertTasks` from `index`/`score` (same shared-expert-as-task-0 convention).
2. If the selected CPU experts require more than one activation-quantization format
   (`expertTypeGroups`, `numasdevice.cpp:1386-1428`), recurses per-format-group and sums
   partials with `AddTo` — same mixed-quant concern as the top-level check in 1.0.
3. Converts/casts `input` into `expandInput` — a **token-major-per-expert-major-gathered**
   buffer: shared expert's rows first, then each routed expert's assigned rows, contiguous per
   expert (`memcpyTasks`, `numasdevice.cpp:1520-1568`), sized to `alignTotalLines = ceil(totalLines/64)*64`.
4. Gate GEMM+SwiGLU fused per-expert per-NUMA-node op (`MultiThreadGemmAndCrossSwigluOp`,
   `numasdevice.cpp:1611-1619`), striped across NUMA-node shards of the weight
   (`weights[e*2]->numasData[nid]`) and dispatched via `DynamicScheduleTasks(ops)` (work
   distributed across the `AliveThreadPool`, one op-list per NUMA node).
5. Down GEMM likewise (`MultiThreadGemmOp`).
6. **Reduce**: `MultiThreadReduceBatch` (`numasdevice.cpp:1779-1788`) scatters each expert's
   per-row down-projection output back to its original token row, **weighted by that task's gate
   score** (`task_weights[reduceOffset] = weight` from `expertTasks[e][i].second`), accumulating
   up to `k` experts per row (`k = max over rows of #selected-experts-among-cpuExperts`) — this
   is the CPU-side equivalent of the GPU's `FastllmCudaPickOutput` scaled scatter-add.
7. Result written to `cpuOutputBuffer` if non-null (pinned buffer, hybrid path), else
   `output.cpuData` directly (CPU-only path).

---

## 2. Marlin INT4 GEMM kernel

File: `src/devices/cuda/linear/fastllm-marlin.cu` (2672 lines) — a vendored/adapted copy of
vLLM's Apache-2.0 GPTQ-Marlin kernel (`csrc/quantization/gptq_marlin/gptq_marlin{,_repack}.cu`,
credited to Neural Magic / Elias Frantar / IST-DASLab). **This file is self-contained**: it
`#include`s only `fastllm-cuda.cuh` (for a few macros; see §7) plus bare CUDA headers
(`cuda.h`, `cuda_bf16.h`, `cuda_fp16.h`, `cuda_runtime.h`), defines its own local
`TORCH_CHECK` macro (`fastllm-marlin.cu:27-34`, just an `abort()`-on-failure printf — **no
libtorch dependency**), and never calls back into fastllm's global CUDA allocator, cuBLAS
handle, or `fastllm::Data`. All public entries are `extern "C"` and take raw device pointers.

### 2.1 GEMM entry point

```cpp
// fastllm-marlin.cu:2644
extern "C" bool FastllmCudaMarlinHalfInt4Gemm(
    const void *a,            // fp16 activations, device ptr, row-major [size_m, size_k]
    const uint32_t *b_q_weight,  // Marlin-repacked INT4 weight, device ptr (see 2.3)
    const void *b_scales,      // fp16 scales, Marlin-permuted layout (see 2.4), device ptr
    const uint32_t *b_zeros,   // packed uint4 zero points, Marlin-permuted+interleaved (see 2.4)
    void *c,                   // fp16 output, device ptr, row-major [size_m, size_n]
    int size_m, int size_n, int size_k,
    int group_size,            // 32 or 128 — quantization group size along size_k
    int *workspace);           // device int buffer, see 2.5
```

Declared identically in `include/devices/cuda/fastllm-cuda.cuh:231-234`.

Preconditions checked in the function itself (return `false`, not an abort, on failure —
callers must have a fallback path) at `fastllm-marlin.cu:2648-2658`:
- `FastllmCudaMarlinCurrentDeviceSupported()` — current device compute capability ≥ 7.5
  (`sm_75`+; `fastllm-marlin.cu:2583-2592`, via `cudaDevAttrComputeCapabilityMajor/Minor`).
- `size_m > 0 && size_n > 0 && size_k > 0`.
- `group_size == 32 || group_size == 128`, and `size_k % group_size == 0`.
- `size_n % marlin::min_thread_n(=64) == 0 && size_k % marlin::min_thread_k(=64) == 0`.

Internally calls `marlin::marlin_mm<half>(...)` (`fastllm-marlin.cu:2168`) with
`q_type = vllm::kU4` (plain unsigned 4-bit, zero-point mode — *not* the GPTQ `b8`/`b128`-biased
symmetric variant), `has_act_order=false`, `is_k_full=true`, `has_zp=true`,
`num_groups = size_k/group_size`, `dev` = current device, `stream = 0` (default stream — the
caller is responsible for stream ordering if it needs async behavior; fastllm's own caller in
`fastllm-linear-int4group.cu` just runs it on the default stream), `thread_k=thread_n=-1` (auto
config via `determine_thread_config`, `fastllm-marlin.cu:2119-2148`), `sms=-1` (auto = SM count),
`max_par = marlin::max_par (=16)`, `use_fp32_reduce=false`, `is_zp_float=false`.

Comment at `fastllm-marlin.cu:2150-2152` is explicit about the supported subset: *"FastLLM only
calls this file through FastllmCudaMarlinHalfInt4Gemm: half activations, uint4 weights,
AWQ-style zero points, no act-order, group_size 32/128 (group_blocks 2/8), and fp16
reduction."* Only 4 of the template instantiation macros are compiled in
(`FASTLLM_INT4GROUP_CALL_IF`, `fastllm-marlin.cu:2153-2165,2310-2313`) — a much smaller kernel
surface than vLLM's full Marlin (which also supports int8, act-order/GPTQ permutation, fp8
reduction, etc.). A Python port porting *just* this file gets a smaller, easier-to-audit kernel
than upstream vLLM Marlin.

### 2.2 Repack entry points (weight preprocessing)

```cpp
// fastllm-marlin.cu:2594
extern "C" bool FastllmCudaGptqMarlinRepackBitsStream(
    const uint32_t *b_q_weight, uint32_t *out,
    int size_k, int size_n, int num_bits, void *streamPtr);
// fastllm-marlin.cu:2627
extern "C" bool FastllmCudaGptqMarlinRepackStream(
    const uint32_t *b_q_weight, uint32_t *out, int size_k, int size_n, void *streamPtr);
    // = RepackBitsStream(..., num_bits=4, streamPtr)
// fastllm-marlin.cu:2632
extern "C" bool FastllmCudaGptqMarlinRepackBits(
    const uint32_t *b_q_weight, uint32_t *out, int size_k, int size_n, int num_bits);
    // = RepackBitsStream(..., stream=nullptr/default)
// fastllm-marlin.cu:2637
extern "C" bool FastllmCudaGptqMarlinRepack(
    const uint32_t *b_q_weight, uint32_t *out, int size_k, int size_n);
    // = RepackBitsStream(..., num_bits=4, stream=nullptr)
```

Input `b_q_weight` is the **standard GPTQ-style packed layout**: `uint32` words, 8×4-bit values
packed per word, laid out `[size_k/8][size_n]` conceptually (row = group of 8 packed K-values,
column = N) — i.e. this is *not* the raw HF `.qweight` layout directly; fastllm builds this
"standard" packed form itself on-device first (see 2.3) before repacking. Constraints
(`fastllm-marlin.cu:2600-2603`): `num_bits ∈ {4, 8}`, `size_k % tile_k_size == 0`, `size_n %
tile_n_size == 0` (`tile_k_size`/`tile_n_size` are Marlin's internal 16×64 tile constants). No
device-capability check beyond the same sm_75+ gate as the GEMM. Repack runs on `streamPtr`
(pass `nullptr` for the default stream) — this is why the `...Stream` variants exist: repacking
can be launched async and only needs to complete before the first GEMM call that uses `out`.

Repack shuffles/interleaves the raw INT4 values into the tiled thread-register layout Marlin's
main GEMM kernel expects — it must be run once per weight matrix, ahead of time (typically at
model-load time), not per forward call.

### 2.3 Concrete producer pipeline (from `fastllm-linear-int4group.cu`)

Since the Marlin file only deals in the abstract "standard GPTQ packed" format, the *complete*
weight-preparation pipeline (needed to reproduce in Python/NumPy) lives in
`src/devices/cuda/linear/fastllm-linear-int4group.cu`, function
`FastllmCudaInt4GroupEnsureMarlinOnDevice` (`fastllm-linear-int4group.cu:950-1012`), triggered
lazily the first time a given INT4-group weight is used with Marlin eligible shapes. Steps:

1. fastllm's native INT4-group weight storage is nibble-packed row-major `[k][m/2]` bytes (`k`
   = output features, `m` = input features; note this is the opposite convention from GEMM
   `size_n`/`size_k` — see 2.6). A device kernel `FastllmCudaInt4GroupToMarlinQWeightKernel`
   (`fastllm-linear-int4group.cu:850-871`) unpacks that into the "standard" GPTQ `uint32`
   8-per-word layout (`stdQWeight`, `[m/8][k]` conceptually — i.e. transposed relative to the
   nibble-packed source, packing along `m` this time).
2. `FastllmCudaGptqMarlinRepack(stdQWeight, marlinQWeight, m, k)` — note argument order is
   `(size_k=m, size_n=k)` here, i.e. Marlin's "K" is fastllm's input-feature dim `m` and Marlin's
   "N" is fastllm's output-feature dim `k` (see 2.6 for the full convention table).
3. Original INT4 CUDA buffer is freed (`FastllmCudaInt4GroupReleaseOriginalWeight`) and the CUDA
   allocator's idle pool is trimmed (`FastllmCudaClearBigBuffer`) — repacking transiently needs
   source+intermediate+final buffers simultaneously, and the comment at
   `fastllm-linear-int4group.cu:978-984` flags this as a real memory-pressure concern when many
   MoE experts are lazily Marlin-ized.
4. `FastllmBuildMarlinPermutedScalesAndZeros` (`fastllm-linear-int4group.cu:890-948`) builds the
   scale/zero-point tensors in Marlin's required permuted layout (see 2.4) **on the host**, then
   uploads them.
5. A `workspace` int buffer of size `max(1, (k/64)*16)` ints is allocated and zeroed
   (`fastllm-linear-int4group.cu:996-998`) — see 2.5.

### 2.4 Scale / zero-point layout

Built by `FastllmBuildMarlinPermutedScalesAndZeros(weight, scales, zeros, m, k)`
(`fastllm-linear-int4group.cu:890-948`), given `weight.group` (number of quant groups along
`m`), `weight.scales`/`weight.mins`/`weight.zeros` (fastllm's native per-`(output_row,
group)`-indexed float arrays, row-major `[k][group]`, i.e. `src = out*group + g`):

1. Transpose to `[group][k]` (`scaleGN[g*k+out] = weight.scales[out*group+g]`), computing a
   4-bit zero point per `(group, out)` either directly from `weight.zeros` if present, else from
   `round(-min/scale)` clamped to `[0,15]`.
2. Apply Marlin's fixed 64-element column permutation `scalePerm[64]` (`fastllm-linear-int4group.cu:898-907`,
   an interleave pattern `{0,8,16,...,56, 1,9,...,57, ...}`) to every contiguous 64-element chunk
   of the flattened `[group*k]` array — this is Marlin's tensor-core-friendly scale-tile layout,
   identical to vLLM's.
3. `scales[]` becomes `half` (fp16) after permutation — this is the exact array passed as
   `b_scales` to `FastllmCudaMarlinHalfInt4Gemm`. Its logical shape is `[num_groups, size_n]`
   (`num_groups = size_k/group_size`), permuted as above.
4. Zero points: apply an 8-element interleave (`zpInterleave[8] = {0,2,4,6,1,3,5,7}`) then pack 8
   4-bit values per `uint32` (`fastllm-linear-int4group.cu:935-947`) — this is `b_zeros`,
   logically `[num_groups, size_n/8]` `uint32` words.

Python-port takeaway: reproducing Marlin's scale/zero layout requires exactly these two
permutations (`scalePerm` 64-lane shuffle, `zpInterleave` 8-lane shuffle + 4-bit pack) — they are
small, fixed, hardware-tile-derived tables, safe to hardcode in a NumPy preprocessing routine
rather than re-deriving.

### 2.5 Workspace

`workspace` is a plain `int32` device buffer used internally by Marlin's GEMM as a set of
**mutex-like reduction locks** (`int* locks = (int*)workspace;`, `fastllm-marlin.cu:2276`) for
its split-K/parallel-M reduction scheme; it must be **zeroed** before each independent GEMM
"session" that doesn't reuse a previous call's in-flight lock state (fastllm zeroes it once at
weight-prep time via `FastllmCudaMemset0`, `fastllm-linear-int4group.cu:998`, then reuses the
same workspace buffer across forward calls without re-zeroing — Marlin's locking scheme is
self-resetting per call). Required size used by fastllm: `max(1, (size_k/64) * 16) * sizeof(int)`
bytes (`fastllm-linear-int4group.cu:996`) — this is a safe-upper-bound sizing rather than an
exact formula from the Marlin kernel itself; one workspace buffer is cached per weight tensor
(`weight.extraCudaData[INT4GROUP_MARLIN_WORKSPACE_IDX]`).

### 2.6 Shape/argument convention at the real call site

`FastllmCudaHalfMatMulFloatInt4Group(input, weight, bias, output, n, m, k)`
(`fastllm-linear-int4group.cu:1679`) calls (`fastllm-linear-int4group.cu:1721-1722`):

```cpp
FastllmCudaMarlinHalfInt4Gemm(cudaInput, marlinQWeight, marlinScales, marlinZeros,
                              cudaOutput, /*size_m=*/n, /*size_n=*/k, /*size_k=*/m,
                              groupCnt, marlinWorkspace);
```

I.e. fastllm's own `(n, m, k)` convention is `n`=batch rows, `m`=input features (GEMM K),
`k`=output features (GEMM N) — so when mapping onto Marlin's `(size_m, size_n, size_k)` naming,
**Marlin's `size_n` is fastllm's `k` and Marlin's `size_k` is fastllm's `m`**. Any Python wrapper
should use fastllm's `(n, m, k)` naming to match the rest of the codebase and translate
internally.

### 2.7 Eligibility gate for using Marlin at all

`FastllmCudaInt4GroupMarlinEnabled(n, m, k, groupCnt)` (`fastllm-linear-int4group.cu:873-888`):
requires sm_75+, `n >= 1`, `groupCnt ∈ {32, 128}`, `m % groupCnt == 0`, `groupCnt % 16 == 0`,
`m % 64 == 0`, `k % 64 == 0` (`CUDA_NO_TENSOR_CORE` build disables Marlin outright). It is also
explicitly **disabled for routed MoE expert weights**
(`fastllm-linear-int4group.cu:1686-1687`, matched by tensor name containing `.mlp.experts.` or
`.block_sparse_moe.experts.`) — those go through a separate fused batch-1 MoE INT4-group kernel
instead, because converting one routed expert to Marlin format would invalidate the device-side
expert-pointer table the fused MoE kernel depends on. Marlin is used for **dense** linear layers
(attention QKVO, MLP gate/up/down of dense layers, and the *shared* MoE expert) only.

---

## 3. FP8 block-scaled GEMV kernel

File: `src/devices/cuda/linear/fastllm-linear-fp8.cu`. Two distinct FP8-E4M3 "block" weight
formats exist in the codebase; the one named in the task (`FastllmQuantizeLinearWeightFP8E4M3Block128Kernel`)
produces the canonical `DataType::FP8_E4M3_BLOCK_128` storage format used everywhere in fastllm
for this dtype (confirmed by `Data::GetDataBytes`, `src/fastllm.cpp:575-577`, using the identical
byte formula — see 3.2). A second, generic `DataType::FP8_E4M3` format with a *separate* scale
tensor and configurable `blockM`/`blockK` also exists (§3.4) — do not conflate the two.

### 3.1 Quantize kernel (weight → FP8_E4M3_BLOCK_128)

```cpp
// fastllm-linear-fp8.cu:41 (templated on T = half or __nv_bfloat16)
template <typename T>
__global__ void FastllmQuantizeLinearWeightFP8E4M3Block128Kernel(
    const T *input, uint8_t *output, int rows, int columns,
    int packedRowBytes, int blocksPerRow, int totalBlocks);
```

Host entry:

```cpp
// fastllm-linear-fp8.cu:87 ; declared include/devices/cuda/fastllm-cuda.cuh:522
bool FastllmCudaQuantizeLinearWeightFP8E4M3Block128(
    const fastllm::Data &input, fastllm::Data &output);
```

Preconditions (`fastllm-linear-fp8.cu:89-97`): `input` must be CUDA-resident, 2-D
`[rows, columns]`, `columns % 128 == 0`, dtype FLOAT16 or BFLOAT16. Sets `output.dataType =
FP8_E4M3_BLOCK_128`, resizes/allocates it, then launches the kernel with
`grid = min(totalBlocks, SM_count*8)`, `256` threads/block (8 warps/block, one warp per 128-value
tile, grid-stride loop over `totalBlocks = rows * (columns/128)` tiles).

**Semantics** (`fastllm-linear-fp8.cu:50-84`): block size is **128 along the column
(reduction/input-feature) dimension**, independently per row. Each warp handles one
`(row, 128-column-block)` tile: loads the 128 values (4 per lane via the 32-lane warp), computes
`localMax = max(|x|)` via warp-shuffle reduction, `scale = localMax / 448.0f` (448 = FP8 E4M3's
max representable magnitude; `scale=1.0` if `localMax<=0` to avoid div-by-zero), then writes each
value as `__nv_cvt_float_to_fp8(x * (1/scale), __NV_SATFINITE, __NV_E4M3)` — genuine hardware FP8
E4M3 (1 sign + 4 exponent + 3 mantissa bits), **not** a plain int8 quantization (a Python NumPy
reference implementation must emulate E4M3 rounding/saturation, not just clip-and-cast to int8 —
this is a correctness-relevant gotcha for the port).

### 3.2 Data layout

`packedRowBytes = columns + blocksPerRow * sizeof(float)` where `blocksPerRow = columns / 128`.
Physical layout per row (`fastllm-linear-fp8.cu:71-75`, confirmed by the *consumer* kernel's
comment at `fastllm-linear-fp8.cu:633`):

```
row 0: [fp8_0 .. fp8_127][float32 scale_0][fp8_128 .. fp8_255][float32 scale_1] ... (blocksPerRow times)
row 1: <same, packedRowBytes further into the buffer>
...
```

I.e. **the scale is stored inline, immediately after each 128-value block, interleaved with the
data — not in a separate tensor.** `Data::GetDataBytes(FP8_E4M3_BLOCK_128, rows, columns)`
(`src/fastllm.cpp:575-577`) computes exactly `rows * (columns + ceil(columns/128)*sizeof(float))`,
confirming this is the canonical on-disk/on-device representation for this dtype system-wide
(also true for weights loaded directly from a pre-quantized checkpoint, not just weights produced
by this on-the-fly quantizer).

### 3.3 GEMV consumer kernels and launcher

Two kernel variants consume this format:

```cpp
// fastllm-linear-fp8.cu:613 — one CUDA block per output row, shared-mem reduction
template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow(
    half *A, uint8_t *B, half *C, half *bias, int m, int k, int perRow);

// fastllm-linear-fp8.cu:706 — one warp per output row, warp-shuffle reduction (faster)
template <int WARPS_PER_BLOCK, int PART>
__global__ void FastllmGemvHalfFP8E4M3Block128KernelWarpMultiRow(
    const half *A, const uint8_t *B, half *C, const half *bias, int m, int k, int perRow);
```
`perRow` = `packedRowBytes` from §3.2 (bytes per weight row, including inline scales).
`PART` = number of activation rows (batch) processed per invocation (register-blocked, up to 8);
`A` is `[PART, m]` row-major fp16 activations, `C` is `[PART, k]` row-major fp16 output (`C[row +
k*x]` addressing — output is *column-major across the PART batch* in the sense that consecutive
batch elements are `k` floats apart, but each is contiguous over `k`). There is also a bfloat16
mirror pair, `FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow` / `...WarpMultiRow`
(`fastllm-linear-fp8.cu:1585, 1674`).

Launchers (choose kernel variant + batch-blocking factor):
```cpp
// fastllm-linear-fp8.cu:777
void LaunchFastllmGemmFp16FP8E4M3Block128(half *input, uint8_t *weight, half *output,
                                           half *bias, int n, int m, int k, int perRow);
// fastllm-linear-fp8.cu:1751
void LaunchFastllmGemmBF16FP8E4M3Block128(__nv_bfloat16 *input, uint8_t *weight,
                                           __nv_bfloat16 *output, __nv_bfloat16 *bias,
                                           int n, int m, int k, int perRow);
```
These take **raw device pointers only** — no `fastllm::Data` dependency — and are the cleanest
extraction targets for a standalone `.so` (see §7). They are declared with external C++ linkage
in the `.cu` file but not exposed in any header, so a wrapper TU must forward-declare them with
matching signatures to call them directly, or new `extern "C"` shims should be added.

`fastllm::Data`-based entry points that wrap all of the above for full-model use (n≥32 batches
fall back to a cuBLAS dequant-then-GEMM path instead of the direct GEMV kernel — see §7 for why
that path pulls in global state):
```cpp
bool FastllmCudaMatMulFloatFP8E4M3Block128(const Data&, Data& weight, const Data& bias, Data& output, int n, int m, int k); // fastllm-linear-fp8.cu:872
bool FastllmCudaHalfMatMulFloatFP8E4M3Block128(const Data&, Data& weight, const Data& bias, Data& output, int n, int m, int k); // :1441
bool FastllmCudaBFloat16MatMulFP8E4M3Block128(const Data&, Data& weight, const Data& bias, Data& output, int n, int m, int k); // :1786
bool FastllmCudaHalfMatMulFloatFP8E4M3Block128Swiglu(...);   // fused gate+up SwiGLU variant
bool FastllmCudaHalfMatMulFloatFP8E4M3Block128AddTo(...);    // fused +=alpha* variant
```
(All declared in `include/devices/cuda/fastllm-cuda.cuh:741-829`.)

MoE-specific batch-1 indexed variants also exist and consume the same per-row block128 layout
per selected expert weight, e.g. `FastllmCudaHalfMergeMOEFP8E4M3Block128Batch1Indexed`
(`fastllm-cuda.cuh:703-710`) / BFloat16 counterpart (`:767-774`) and a fused
`FastllmCudaHalfFusedMOEFP8E4M3Block128` (`:711`) — used by `DoCudaMergeMOE`'s batch-1 decode
dispatch (`cudadevice.cpp:5674-5677`) for FP8-quantized MoE expert weights.

### 3.4 The other FP8 format, for contrast (do not confuse with 3.1-3.3)

Generic `DataType::FP8_E4M3` weights (no `_BLOCK_128` suffix) store **plain** FP8 bytes,
`[k, m]` row-major, no inline scale — the scale lives in a **separate** `float32` array of shape
`[ceil(k/blockK), ceil(m/blockM)]` (row-major, `weight.blockK`/`weight.blockM` are per-tensor
fields on `Data`, set at load time — typically 128×128 to match DeepSeek-native
`weight_scale_inv` checkpoints, but not hardcoded to 128). Consumer:
`FastllmGemvHalfFP8E4M3KernelWarpMultiRowBlock128` (`fastllm-linear-fp8.cu:427`, note: *no*
`perRow`/interleave — it reads `scales[((row)>>7)*ms + col_block]` from the separate array,
`ms=m/128`) is selected only when `blockM==128 && blockK==128 && m%128==0`
(`fastllm-linear-fp8.cu:572`); non-128 block shapes fall back to the generic
`FastllmGemvHalfFP8E4M3KernelWarpMultiRow` kernel (blockM/blockK passed as runtime args, not
compiled-in constants). This is the format used for regular dense/MoE model weights loaded
directly from a DeepSeek-style FP8 safetensors checkpoint; §3.1-3.3's interleaved
`FP8_E4M3_BLOCK_128` format is, in this codebase, used specifically as an **on-the-fly weight
requantization** target (its only call site is `Qwen3_5Model::PrepareMtpDraftLmHeadWeights`,
`src/models/qwen3_5.cpp:16888-16951`, which requantizes the `lm_head.weight` from FP16/BF16 into
FP8 for a multi-token-prediction draft head). A Python port targeting "the FP8 GEMV kernel" as
generally used for model weights should implement **both**: the separate-scale-tensor format
(§3.4) for main-model FP8 weights, and the interleaved format (§3.1-3.3) only if it also wants to
reproduce the MTP draft-head fast-quantization path.

---

## 4. CUDA slab allocator (`--cuda_slab`)

Purpose: avoid CUDA driver overhead (and possible fragmentation) from allocating **many small
per-expert weight buffers** individually (each MoE layer can have 64-256+ experts, each with 2+
weight tensors) by bump-allocating them out of a small number of large, contiguous arena
allocations ("slabs").

### 4.1 Enable/configure

- C API: `void SetCudaSlabMB(int mb)` / `int GetCudaSlabMB()` (`include/fastllm.h:79-80`, impl
  `src/fastllm.cpp:378-387`) — sets a process-global `cudaSlabMB` and calls
  `FastllmCudaSetWeightSlabBytes((size_t)mb * 1024 * 1024)`.
- Python binding: `tools/fastllm_pytools/llm.py:492-493`,
  `set_cuda_slab(mb)` → `fastllm_lib.set_cuda_slab(ctypes.c_int(mb))`
  (native export `tools/src/pytools.cpp:61-62`).
- CLI: `--cuda_slab` int arg (`tools/fastllm_pytools/util.py:237`, default `0` = off); the
  higher-level launcher script defaults it to **256 MB** whenever unset (`util.py:448-449`,
  `if args.cuda_slab <= 0: args.cuda_slab = 256`), then calls `llm.set_cuda_slab(args.cuda_slab)`
  (`util.py:566-567`) — i.e. in practice the slab allocator is on by default at 256 MB per slab
  unless a user explicitly passes `--cuda_slab 0`.

### 4.2 Mechanism

State (`src/devices/cuda/fastllm-cuda.cu:2919-2938`):
```cpp
struct FastllmCudaWeightSlab { void *base; size_t size, used; int activeBlocks; };
struct FastllmCudaWeightSlabPtr { int device; void *base; };
static std::atomic<size_t> fastllmCudaWeightSlabBytes;           // the configured slab size
static std::map<int, std::vector<FastllmCudaWeightSlab>> fastllmCudaWeightSlabs;   // per device id
static std::map<void*, FastllmCudaWeightSlabPtr> fastllmCudaWeightSlabPtrs;         // sub-alloc → owning slab
```

Allocation entry: `void *FastllmCudaMallocModelWeight(size_t size)`
(`src/devices/cuda/fastllm-cuda.cu:2952-3000`):
1. If slab size is `0`, or `size == 0`, or `size > slabBytes/2` (a single weight tensor bigger
   than half a slab isn't worth slab-packing — falls back to the normal pooled
   `FastllmCudaMalloc(size)`), skip slabbing entirely.
2. Else, under `fastllmCudaWeightSlabMutex`, linear-scan the current device's slab list for one
   with `size - used >= aligned` (`aligned = round_up(size, 256)`); if none fits, `cudaMalloc` a
   brand-new `slabBytes`-sized slab and append it.
3. Bump-allocate: `ret = slab.base + slab.used; slab.used += aligned; slab.activeBlocks++;`
   record `ret → {device, slab.base}` in `fastllmCudaWeightSlabPtrs`.

**Granularity**: fixed slab size = the configured `--cuda_slab` MB value (default effectively 256
MB when launched via the standard tooling), sub-allocations aligned to **256 bytes**, and a slab
is only ever grown by allocating an entirely new slab of the same fixed size (no
splitting/coalescing/realloc within a slab — this is a pure bump allocator, never compacts).

Freeing: `FastllmCudaTryFreeWeightSlabPtr(void *ret)`
(`src/devices/cuda/fastllm-cuda.cu:3002-3041`) looks up `ret` in `fastllmCudaWeightSlabPtrs`;
if found, decrements that slab's `activeBlocks`, and only when it hits `0` does the *entire slab*
get `cudaFree`'d and erased — i.e. **individual sub-allocations are never actually freed**, only
logically released; physical memory is reclaimed only when every weight ever carved out of a
given slab has been released. This is called transparently from the normal free path
(`fastllm-cuda.cu:3705, 3791` try the slab-free first, falling through to a normal `cudaFree`/pool
free if the pointer wasn't slab-allocated).

### 4.3 What actually uses the slab

Only tensors flagged `isModelWeight && !directMemory` route through
`FastllmCudaMallocModelWeight` — see `CudaMallocForData` (`src/fastllm.cpp:107-112`):
```cpp
static void *CudaMallocForData(const Data &data, uint64_t bytes) {
    if (data.isModelWeight && !data.directMemory) return FastllmCudaMallocModelWeight(bytes);
    return data.directMemory ? FastllmCudaDirectMalloc(bytes) : FastllmCudaMalloc(bytes);
}
```
Activations, KV cache, and scratch/temporary buffers always go through the ordinary pooled
allocator (`FastllmCudaMalloc`/`FastllmCudaFree`, a separate big-buffer/idle-list pool elsewhere
in `fastllm-cuda.cu`), never the slab. Python-port equivalent: a simple per-device bump allocator
(e.g. backed by one or more `cudaMalloc`'d/`cp.cuda.Memory`-backed arenas) used *only* for static
expert weight tensors at load time, with the same "whole slab freed only when refcount hits
zero" semantics if per-expert eviction/reload is needed (e.g. an LRU expert cache, as sketched in
`fastllm-port-brief.md`).

---

## 5. `MoeExpertSpeedEstimator` — already covered in full in §1.2

(Repeated pointer per the task's item numbering: see "1.2 Dynamic threshold benchmark" above for
the complete class description — shape-keyed profile cache, geometric row-count sampling,
CPU/GPU wall-clock micro-benchmarking of one representative expert, linear interpolation, and the
`argmin |cpuTime(t) - gpuTime(t)|` threshold search.)

---

## 6. Stream/thread orchestration and pinned-memory DMA — summary

(Full detail in §1.3-1.5; this is the condensed cross-reference the task asked for.)

- **GPU experts run on a background `std::thread`** (`numasdevice.cpp:2564-2569`), *not* just a
  separate CUDA stream — the thread itself calls `FastllmCudaSetDevice(gpuId)` then
  `DoCudaMergeMOEFromCPU`, which internally uses the **default stream** for compute and one
  **private non-blocking stream** (`copyStream`, created fresh each call via
  `FastllmCudaStreamCreate(true)`, `cudadevice.cpp:5184`) purely for prefetching each next
  expert's weights while the current expert computes on the default stream — synchronized via a
  CUDA event (`computeDoneEvent`), not host-side blocking, so weight eviction never races
  in-flight compute (`cudadevice.cpp:5304-5318`).
- **CPU experts run on the calling thread** via the NUMA-aware thread pool
  (`GetAlivePool()`/`DynamicScheduleTasks`) — a separate, CPU-only worker-thread pool, unrelated
  to the GPU thread above.
- **Synchronization point**: `gpuThread.join()` (`numasdevice.cpp:2599`) — plain thread join, no
  CUDA-level cross-stream event is used to gate the CPU thread on the GPU thread (the GPU thread
  internally waits on its own streams before returning).
- **CPU→GPU partial-result handoff** uses a 3-hop pinned-memory pipeline, fully described in
  §1.3-1.4:
  1. CPU expert loop writes its score-weighted partial sum **directly into a page-locked host
     buffer** (`FastllmCudaHostMalloc`-backed, cached per-layer-parity in
     `FastllmMoeDataManagerNumas::pinnedOutput`) instead of into `output.cpuData` — i.e. no extra
     host-side memcpy, the reduce kernel's final write target *is* the pinned buffer
     (`cpuOutputBuffer` param threaded all the way through `DoNumasMergeMOEOnCPU`).
  2. `FastllmCudaCopyFromPinnedHostToDeviceAsync(gpuStagingBuffer, pinnedBuffer, bytes,
     copyStream)` (`numasdevice.cpp:2590-2592`) — async H2D copy on a **third** dedicated stream
     (`FastllmMoeDataManagerNumas::gpuOutputCopyStream`, distinct from the GPU thread's internal
     `copyStream`), started as soon as the CPU partial is ready, potentially still while the GPU
     thread is mid-flight.
  3. After `gpuThread.join()`, `FastllmCudaStreamSynchronize(cpuOutputCopyStream)` then a
     device-side `FastllmCudaAddTo(gpuOutputAlias, cpuStagingAlias, 1.0f)` combines the two
     partials in-place on `output.cudaData` (`numasdevice.cpp:2601-2606`).
- All of the pinned host buffer, the GPU-side staging buffer, and the copy stream are **cached
  and reused** across forward calls (grow-only, per `layer%2` bucket) — no allocation happens on
  steady-state forward passes, only on the first call or a size increase.

---

## 7. Files needed for a standalone `.so` (Marlin GEMM + FP8 GEMV + quantize/repack only)

### 7.1 Marlin — genuinely self-contained

**Compile:** `src/devices/cuda/linear/fastllm-marlin.cu` alone.

**Include chain:** `#include "fastllm-cuda.cuh"` (`include/devices/cuda/fastllm-cuda.cuh`) →
`#include "fastllm.h"` (`include/fastllm.h`) → `devices/cpu/alivethreadpool.h`,
`third_party/json11/json11.hpp`, and (only if `USE_SENTENCEPIECE` is defined)
`sentencepiece_processor.h`. **None of these transitively-included declarations are actually
used** by `fastllm-marlin.cu` itself (verified: the file calls no `fastllm::Data` method, no
`FastllmCuda*` helper from `fastllm-cuda.cu`, nothing from the thread pool) — the include exists
only because the project's convention is every `.cu` under `devices/cuda` includes the common
header. **For a minimal standalone build, this include can be dropped or replaced** with just the
bare CUDA headers (`cuda.h`, `cuda_fp16.h`, `cuda_bf16.h`, `cuda_runtime.h`) that
`fastllm-marlin.cu` already includes directly — no other project header is structurally required.
It should link with just `-lcudart` (no cuBLAS, no cuBLASLt, no CUTLASS).

**Global state:** none. Every exported function (`FastllmCudaMarlinHalfInt4Gemm`,
`FastllmCudaGptqMarlinRepack{,Bits,Stream,BitsStream}`) takes all its state as
arguments/pointers, allocates nothing internally (the `workspace` buffer is caller-provided), and
uses the default stream unless a stream pointer is passed in explicitly (repack functions only;
the GEMM itself is hardcoded to stream `0`, `fastllm-marlin.cu:2662-2667` passes literal `0`/`dev`
to `marlin_mm`, not a stream parameter — this means **the GEMM call is not currently
stream-parameterizable in fastllm's copy of this file**; a Python port wanting async/multi-stream
Marlin GEMM would need to add a `cudaStream_t` parameter to `FastllmCudaMarlinHalfInt4Gemm` and
thread it through to `marlin_mm`'s `stream` argument, which the underlying template already
supports).

**Device-side prerequisite state (not global, but must exist before calling)**: the weight
tensor must already be Marlin-repacked (§2.2-2.3) and its scales/zeros already permuted (§2.4) —
this repacking is a one-time offline/load-time step, not part of the GEMM call itself.

### 7.2 FP8 block128 GEMV — needs new thin wrappers, not the existing public entry points

**Do not link the existing `Fastllm{Cuda,CudaHalf,CudaBFloat16}MatMulFloatFP8E4M3Block128`
functions as-is** if the goal is a minimal `.so`. Reasons:
- They take `fastllm::Data&` (input/weight/bias/output), so calling them requires constructing
  real `fastllm::Data` objects — pulling in all of `src/fastllm.cpp` (weight
  loading/serialization, tokenizer glue, etc.) just to get a thin wrapper class.
- Their `n ≥ 32`/`n ≥ 16` batch fallback paths (`fastllm-linear-fp8.cu:1152, 904`) dequantize to
  fp16 and call `cublasGemmEx` via `getFastllmCublasHandle()` — a **process-global, per-device,
  lazily-created cuBLAS handle** cached in `s_fastllmCublasHandleMap`
  (`src/devices/cuda/fastllm-cuda.cu:309-330`, guarded by `s_fastllmCublasHandleMapMutex`, and
  rebound to `cudaStreamPerThread` on every fetch) — defined in the monolithic
  `src/devices/cuda/fastllm-cuda.cu` (12000+ lines), which itself needs the whole CUDA memory
  pool (`FastllmCudaMalloc`/`FastllmCudaFree`, big-buffer pool, slab allocator from §4,
  `FastllmCudaPrepareInput`/`FastllmCudaPrepareOutput` staging helpers, `FastllmBorrowDequantScratch`
  which borrows a lazily-allocated **FlashInfer workspace buffer** shared with attention
  kernels) to link at all.
- Bias handling caches a device bias buffer keyed on `weight.extraCudaData`/`extraCudaHalfData`
  (per-`Data`-object state, e.g. `FastllmCudaFP8E4M3Block128EnsureHalfBiasOnDevice`,
  `fastllm-linear-fp8.cu:815`) — again tied to the `Data` object lifecycle.

**Recommended extraction** (raw-pointer, no global state, no cuBLAS):
1. Copy out (or `#include` from a trimmed source) just:
   - `FastllmQuantizeLinearWeightFP8E4M3Block128Kernel<T>` (kernel, `fastllm-linear-fp8.cu:41`)
     and a new tiny non-`Data` launcher around it (the existing
     `FastllmCudaQuantizeLinearWeightFP8E4M3Block128` host function is a thin, `Data`-free-logic
     wrapper already — everything after the precondition checks and `output.Allocate()` is just
     `cudaGetDeviceProperties` + the kernel launch, so it's easy to re-derive a raw-pointer
     version: caller supplies `rows, columns, packedRowBytes, blocksPerRow, totalBlocks` and
     pre-allocated device buffers).
   - `FastllmGemvHalfFP8E4M3Block128KernelWarpMultiRow<W,PART>` /
     `FastllmGemvBF16FP8E4M3Block128KernelWarpMultiRow<W,PART>` (kernels) plus their existing
     raw-pointer launchers `LaunchFastllmGemmFp16FP8E4M3Block128`
     (`fastllm-linear-fp8.cu:777`) / `LaunchFastllmGemmBF16FP8E4M3Block128` (`:1751`) — these
     **already have no `fastllm::Data` or global-state dependency**; only need `extern "C"` shims
     with fixed (non-templated) signatures for ctypes.
2. Skip the `n≥16`/`n≥32` cuBLAS-dequant fallback entirely for a from-scratch Python port —
   Python already has cuBLAS/cuBLASLt access via CuPy/PyTorch for large-batch dense GEMM; only
   the small-batch (decode, `n` single-digit) direct FP8 GEMV kernel is the part worth porting
   as a hand-tuned `.so`, matching the `fastllm-port-brief.md` rationale ("Python's only role
   should be calling them").
3. `#include`s needed: `<cuda_fp8.h>`, `<cuda_fp16.h>` (implied via `<cuda_bf16.h>`/half
   support), `<cuda_bf16.h>`, `<cuda_runtime.h>` — no project headers required if the kernels are
   copied into a fresh file with raw signatures.

**Global state for the recommended (raw-pointer) extraction: none** — device selection
(`cudaSetDevice`), stream choice, and all memory ownership are left to the Python caller
(CuPy/ctypes), exactly like the Marlin recommendation in §7.1.

### 7.3 Summary table

| Component | Self-contained as-is? | File(s) | External deps if used as-is | Global state if used as-is |
|---|---|---|---|---|
| Marlin GEMM (`FastllmCudaMarlinHalfInt4Gemm`) | Yes | `fastllm-marlin.cu` | CUDA runtime only | none |
| Marlin repack (`FastllmCudaGptqMarlinRepack*`) | Yes | `fastllm-marlin.cu` | CUDA runtime only | none |
| FP8 block128 quantize kernel | Kernel: yes. `Data`-based host fn: no | `fastllm-linear-fp8.cu` | Kernel: none. Host fn: `fastllm::Data` (`fastllm.cpp`) | Host fn: CUDA mem pool (`fastllm-cuda.cu`) |
| FP8 block128 GEMV (small-batch) | Launcher: yes. `Data`-based host fn: no | `fastllm-linear-fp8.cu` | Launcher: none. Host fn: `fastllm::Data`, cuBLAS handle, FlashInfer scratch | Host fn (large batch): global cuBLAS handle map, CUDA mem pool, dequant scratch |

---

## Appendix: cross-check against `fastllm-port-brief.md`'s existing sketches

The repo already has a planning doc (`/data/user/ypzhang/dev/github/codemk8/fastllm-py/fastllm-port-brief.md`)
with illustrative Python sketches. Two corrections worth flagging against this report's findings:

- Its `marlin_gemm` ctypes wrapper (brief lines ~399-413) omits the `b_zeros` argument entirely —
  the real kernel is AWQ/GPTQ-zero-point mode (`has_zp=true`, `q_type=kU4`), so `b_zeros` is
  mandatory, not optional (§2.1/2.4).
- Its `quantize_fp8_block128` NumPy sketch (brief lines ~106-120) quantizes to `np.int8` via
  simple clip-and-cast. The real kernel casts to genuine **FP8 E4M3** via
  `__nv_cvt_float_to_fp8(..., __NV_E4M3)` (§3.1) — a NumPy port needs an E4M3 emulation (4
  exponent bits, 3 mantissa bits, bias 7, no infinities, saturate at ±448) rather than int8
  quantization, or results will not match fastllm's numerics bit-for-bit and will have
  different rounding/saturation behavior even at similar relative error.
