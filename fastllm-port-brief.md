# Porting fastllm's Inference Strategy to a Python-Native Engine

## Target

Build a Python inference engine (CuPy + NumPy + asyncio) that replicates fastllm's
hybrid CPU/GPU MoE inference strategy while gaining Python's ecosystem advantages.
Target: run DeepSeek V4 (671B) class MoE models on a single
consumer GPU + server RAM at 20-40 tok/s for single-stream decode.
We can of course use small models like Qwen/Qwen3-0.6B to debug

fastllm's source is /data/user/ypzhang/dev/github/codemk8/fastllm-py/fastllm. And it's github url is
https://github.com/ztxz16/fastllm

This host has two 4090 nvidia GPUs, feel free to use it for debugging

## What fastllm Does Well — Must Replicate

### 1. Fine-Grained Device-Level Scheduling

```
--device "{'cuda:0':3, 'cuda:1':2}"       → 3/5 layers on GPU0, 2/5 on GPU1
--moe_device "{'cuda':1, 'numa':8, 'disk':1}" → 10% experts GPU, 80% NUMA, 10% disk
--moe_device_layers 8                      → last 8 MoE layers use moe_device, rest on GPU
--cuda_slab 1024                           → slab allocator for many small expert weight allocations
```

Python equivalent: a `DeviceMap` class that maps layer IDs → (device, device) for dense/MoE
and a `MoeExpertRouter` that assigns individual experts to GPU/CPU/NUMA/disk at runtime.

### 2. Hybrid MoE Forward with Concurrent CPU+GPU Execution

The critical loop from `numasdevice.cpp:2532-2610` and `cudadevice.cpp:5184-5327`:

```
1. Gate (GPU): compute routing distribution for hidden states
2. Split: each expert gets a task list of (token_idx, routing_score)
3. Benchmark determines expertLimit threshold:
   - Task count < threshold → CPU (better latency per token)
   - Task count >= threshold → GPU (better throughput for many tokens)
4. Concurrent execution:
   - GPU experts: std::thread runs DoCudaMergeMOEFromCPU() on gpu_stream
   - CPU experts: main thread runs DoNumasMergeMOEOnCPU()
5. Merge: CPU partial result → pinned buffer → async DMA to GPU → element-wise add
```

Python equivalent:
```python
async def moe_forward(x, gate, experts, expert_router):
    scores = gate(x)                                # GPU
    expert_tasks = route(scores, top_k=8)           # CPU/GPU
    cpu_set, gpu_set = expert_router.split(expert_tasks)

    # Launch GPU on separate stream (non-blocking)
    gpu_stream = cp.cuda.Stream(non_blocking=True)
    with gpu_stream:
        gpu_out = compute_experts(x, experts, gpu_set, expert_tasks)

    # CPU experts in thread pool (or main thread if GPU is async)
    cpu_out = await loop.run_in_executor(pool, compute_experts_cpu, x, experts, cpu_set)

    gpu_stream.synchronize()
    result = gpu_out + cpu_out
    result *= scores.sum()  # score-weighted
    return result
```

### 3. Kernel-Level Quantized GEMV

fastllm has three critical kernel families. For Python, the strategy is:

| Kernel | Strategy |
|--------|----------|
| Marlin INT4 GEMM | **Do not rewrite.** Compile `fastllm-marlin.cu` to `.so`, wrap via ctypes |
| FP8 block-scaled GEMV | **Do not rewrite.** Compile `fastllm-linear-fp8.cu` to `.so`, wrap via ctypes |
| Paged attention | Use `flashinfer` PyPI package (already has Python bindings) |
| Simple fused ops (RMSNorm+RoPE+linear) | Write as CuPy RawKernel (~50 lines each) |

Rationale: The Marlin and FP8 GEMV kernels each took months to tune. They are ~5000+ lines
of inline PTX, warp shuffle, vectorized loads, and software prefetch. Python's only role
should be calling them — not reimplementing them.

### 4. Universal Weight Loading with Auto-Quantization

fastllm reads HuggingFace safetensors directly and quantizes on-the-fly. Python can do
this cleaner:

```python
from safetensors import safe_open
import cupy as cp

def load_weights(model_id: str, dtype_map: dict[str, str]):
    """Load and quantize weights in a single pass."""
    weights = {}
    for shard_path in sorted(Path(model_id).glob("*.safetensors")):
        with safe_open(shard_path, framework="np") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                target_dtype = dtype_map.get(key, "float16")
                if target_dtype == "fp8_e4m3":
                    weights[key] = quantize_fp8_block128(tensor)
                elif target_dtype == "int4":
                    weights[key] = quantize_int4_group(tensor)
                else:
                    weights[key] = cp.array(tensor, dtype=cp.float16)
    return weights

def quantize_fp8_block128(x: np.ndarray, block_size: int = 128):
    """Per-block FP8 quantization. Port of FastllmQuantizeLinearWeightFP8E4M3Block128Kernel."""
    orig_shape = x.shape
    x_2d = x.reshape(orig_shape[0], -1)
    n_blocks = x_2d.shape[1] // block_size
    data = np.zeros((x_2d.shape[0], n_blocks, block_size), dtype=np.uint8)
    scales = np.zeros((x_2d.shape[0], n_blocks), dtype=np.float32)
    for i in range(x_2d.shape[0]):
        for b in range(n_blocks):
            block = x_2d[i, b*block_size:(b+1)*block_size]
            scale = np.max(np.abs(block)) / 448.0
            if scale == 0:
                scale = 1.0
            data[i, b] = np.clip(block / scale, -448, 448).astype(np.int8).view(np.uint8)
            scales[i, b] = scale
    return {"data": cp.array(data.reshape(x_2d.shape[0], -1)), "scales": cp.array(scales)}
```

For optimal performance, move quantization to GPU using CuPy elementwise operations.

---

## What Python Adds — Superior to fastllm

### 1. Zero-Cost New Model Support

```python
from transformers import AutoConfig

def build_model(model_id: str, config_override: dict = None):
    """Parse any HF config.json → build compute graph — no per-model C++ needed."""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if config_override:
        for k, v in config_override.items():
            setattr(config, k, v)
    
    return ModelGraph(
        num_layers=config.num_hidden_layers,
        hidden_dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        intermediate_dim=config.intermediate_size,
        num_experts=getattr(config, "num_experts", 0),
        num_experts_per_tok=getattr(config, "num_experts_per_tok", 0),
        rope_theta=config.rope_theta,
        norm_eps=config.rms_norm_eps,
        # Additional fields parsed automatically from config
    )
```

A new model on HuggingFace today → running on this engine in minutes, not months.

### 2. Async Scheduling with Cross-Layer Prefetch

fastllm's scheduler is synchronous per layer. Python's `asyncio` makes cross-layer prefetch natural:

```python
class PrefetchingScheduler:
    def __init__(self, expert_cache: ExpertCache, copy_stream):
        self.cache = expert_cache
        self.copy_stream = copy_stream
    
    async def forward(self, x, layers, expert_router):
        kv_caches = [None] * len(layers)
        
        for i, layer in enumerate(layers):
            # Layer i's attention (GPU) — runs while we prepare layer i+1
            attn_future = asyncio.create_task(
                gpu_attention_async(x, kv_caches[i], layer.attn_weights)
            )
            
            # Prefetch layer i+1's likely experts (based on layer i's gate output
            # or frequency statistics from prior tokens)
            if i + 1 < len(layers):
                likely_experts = predict_next_layer_experts(i + 1, expert_router)
                for expert_id in likely_experts:
                    if expert_id not in self.cache:
                        self.cache.prefetch_async(expert_id, self.copy_stream)
            
            attn_out = await attn_future
            
            # MoE (hybrid GPU/CPU)
            if layer.is_moe:
                x = await moe_forward_async(attn_out, layer)
            else:
                x = mlp_forward(attn_out, layer)
        
        return x

async def moe_forward_async(x, layer: MoELayer):
    """Overlaps GPU expert computation with CPU expert computation."""
    gate_out = await layer.gate_async(x)
    tasks = route_experts(gate_out, layer.top_k)
    cpu_set, gpu_set = layer.expert_router.split(tasks)
    
    gpu_future = asyncio.ensure_future(
        gpu_experts_async(x, layer.expert_cache, gpu_set, tasks)
    )
    cpu_result = await cpu_experts_async(x, layer.weights, cpu_set, tasks)
    gpu_result = await gpu_future
    
    # Async merge: DMA cpu_result from pinned buffer to GPU staging,
    # then element-wise add (can overlap with next operation)
    return gpu_result + cpu_result
```

### 3. Persistent GPU Expert LRU Cache

fastllm allocates and frees GPU buffers for experts per layer per request. Python can maintain
a persistent cache across layers and requests:

```python
class GpuExpertCache:
    def __init__(self, max_bytes: int, slab_allocator=None):
        self.max_bytes = max_bytes
        self.used = 0
        self.cache: dict[int, cp.ndarray] = {}
        self.lru: collections.OrderedDict[int, None] = collections.OrderedDict()
        self.hits = 0
        self.misses = 0
        self.slab = slab_allocator  # use --cuda_slab pattern
    
    def get_or_upload(self, expert_id: int, cpu_weight: np.ndarray, 
                      stream: cp.cuda.Stream) -> cp.ndarray:
        if expert_id in self.cache:
            self.lru.move_to_end(expert_id)
            self.hits += 1
            return self.cache[expert_id]
        
        self.misses += 1
        weight_bytes = cpu_weight.nbytes
        
        # Evict if needed
        while self.used + weight_bytes > self.max_bytes and self.lru:
            victim = self.lru.popitem(last=False)[0]
            self.used -= self.cache[victim].nbytes
            del self.cache[victim]  # CuPy will free GPU memory
        
        # Upload asynchronously
        with stream:
            gpu_weight = cp.empty_like(cpu_weight) if self.slab is None \
                         else self.slab.allocate(cpu_weight.shape, cpu_weight.dtype)
            gpu_weight.set(cpu_weight)  # async memcpy
        
        self.cache[expert_id] = gpu_weight
        self.lru[expert_id] = None
        self.used += weight_bytes
        return gpu_weight
    
    def prefetch_async(self, expert_id: int, cpu_weight: np.ndarray,
                       stream: cp.cuda.Stream):
        """Prefetch without blocking — call during attention."""
        if expert_id not in self.cache:
            self.get_or_upload(expert_id, cpu_weight, stream)
    
    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
```

### 4. Interactive Development and Debugging

```python
# In Jupyter:
import matplotlib.pyplot as plt

# Visualize expert activation distribution
gate_outputs = layer.gate(x)
expert_freq = cp.bincount(cp.argmax(gate_outputs, axis=-1).flatten())
plt.bar(range(len(expert_freq)), expert_freq.get())
plt.title("Expert Activation Distribution (Layer 12)")

# Profile individual ops
%timeit compute_experts_gpu(x, cached_experts, expert_tasks)
# → 145 µs ± 3 µs per call

# Inspect intermediate tensors
attn_out = layer.attention(x)
print(f"Attention output: shape={attn_out.shape}, "
      f"norm={cp.linalg.norm(attn_out):.2f}, nan={cp.isnan(attn_out).any()}")
```

### 5. Use the Python ML Ecosystem Directly

```python
# Tokenizer — works for every model on HF
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Production paged attention (FlashInfer)
import flashinfer

# Structured output constraints (xgrammar)
import xgrammar

# Fast weight loading (hf_transfer, safetensors)
from safetensors import safe_open

# Serving (FastAPI, uvicorn)
from fastapi import FastAPI
import uvicorn

# Speculative decoding (can add as a plug-in)
# Quantization research (GPTQ, AWQ, SmoothQuant all have Python implementations)
# Evaluation harness (lm-eval)
```

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  Python Orchestration Layer (~800 lines)                          │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ Config   │  │ Scheduler    │  │ Device   │  │ ExpertCache │  │
│  │ Parser   │  │ (asyncio)    │  │ Router   │  │ (LRU, GPU)  │  │
│  └──────────┘  └──────────────┘  └──────────┘  └─────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  Python Runtime (~400 lines)                                      │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────────┐  │
│  │ Weight   │  │ Quantizer    │  │ Model Graph Builder        │  │
│  │ Loader   │  │ (FP8, INT4)  │  │ (from HF config dict)      │  │
│  └──────────┘  └──────────────┘  └────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  Native CUDA Kernels (borrowed from fastllm, compiled as .so)     │
│  ┌────────────────┐  ┌───────────────┐  ┌────────────────────┐   │
│  │ Marlin INT4    │  │ FP8 Block     │  │ Attention          │   │
│  │ (fastllm-marlin│  │ GEMV          │  │ (FlashInfer PyPI)  │   │
│  │  .cu → .so)    │  │ (.cu → .so)   │  │                    │   │
│  └────────────────┘  └───────────────┘  └────────────────────┘   │
├──────────────────────────────────────────────────────────────────┤
│  Hardware                                                         │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │ GPU      │  │ CPU (NumPy,  │  │ NUMA/Disk    │                │
│  │ (CuPy)   │  │  threadpool) │  │ (optional)   │                │
│  └──────────┘  └──────────────┘  └──────────────┘                │
└──────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Foundation (Week 1)
Goal: Load a small dense model (Qwen3-0.6B) and run one forward pass.

- [ ] `config_parser.py`: Read HF config.json → `ModelGraph` dataclass
- [ ] `weight_loader.py`: Load safetensors → dict of numpy arrays
- [ ] `quantizer.py`: FP8 block-128 and INT4 group quantization (NumPy, then CuPy)
- [ ] `device_router.py`: `DeviceMap` class with ratio-based layer assignment
- [ ] `model_graph.py`: Build compute graph from parsed config (no per-model code)
- [ ] `kernels/`: CuPy RawKernel implementations for RMSNorm, RoPE, SwiGLU
- [ ] `test_forward.py`: End-to-end forward pass comparison vs HuggingFace

### Phase 2: MoE and Hybrid Execution (Week 2)
Goal: Run DeepSeek V2-Lite with GPU/CPU hybrid MoE.

- [ ] `expert_router.py`: Gate computation + top-k selection + expert task splitting
- [ ] `expert_cache.py`: GPU LRU cache with async upload/eviction
- [ ] `moe_forward.py`: Concurrent CPU+GPU expert execution + async merge
- [ ] `kernels/fp8_gemv.py`: Wrap fastllm's FP8 GEMV kernel via ctypes
- [ ] `kernels/marlin.py`: Wrap fastllm's Marlin INT4 kernel via ctypes
- [ ] `benchmark.py`: `MoeExpertSpeedEstimator` — runtime profiling for split threshold
- [ ] `test_moe.py`: Correctness and performance test vs reference

### Phase 3: Serving and Advanced Features (Week 3)
Goal: Serve DeepSeek V4 at 20+ tok/s via OpenAI-compatible API.

- [ ] `scheduler.py`: Async scheduler with cross-layer expert prefetch
- [ ] `paged_kv.py`: Integrate FlashInfer paged attention (PyPI package)
- [ ] `server.py`: FastAPI server with `/v1/chat/completions` endpoint
- [ ] `speculative.py` (optional): Draft model for speculative decoding
- [ ] `disk_offload.py` (optional): NVMe streaming for cold experts
- [ ] `multi_gpu.py` (optional): Multi-GPU tensor parallel via NCCL/CuPy
- [ ] `benchmark_throughput.py`: Measure tok/s vs batch size, compare vs vLLM/llama.cpp

---

## Key Design Decisions

### 1. Keep the CUDA kernels as `.so` files, not inline strings

The Marlin and FP8 GEMV kernels are 5000+ lines of hand-tuned PTX. Embedding them
as CuPy RawKernel strings makes them unreadable and unmaintainable. Instead:

```python
import ctypes
import cupy as cp

_lib = ctypes.CDLL("./libfastllm_kernels.so")

def marlin_gemm(a: cp.ndarray, b_quant: cp.ndarray, 
                scales: cp.ndarray, workspace: cp.ndarray) -> cp.ndarray:
    """Wrapper around fastllm's Marlin INT4 GEMM kernel."""
    assert a.dtype == cp.float16
    out = cp.empty((a.shape[0], b_quant.shape[1]), dtype=cp.float16)
    _lib.marlin_gemm(
        ctypes.c_void_p(a.data.ptr),
        ctypes.c_void_p(b_quant.data.ptr),
        ctypes.c_void_p(out.data.ptr),
        ctypes.c_void_p(scales.data.ptr),
        ctypes.c_void_p(workspace.data.ptr),
        ctypes.c_int(a.shape[0]),
        ctypes.c_int(b_quant.shape[1]),
        ctypes.c_int(a.shape[1]),
    )
    return out
```

### 2. Use `asyncio` for scheduling, not hand-rolled threads

fastllm's scheduler is `std::thread` + `std::mutex` + `std::condition_variable`. In
Python, `asyncio` + `concurrent.futures.ThreadPoolExecutor` is simpler and more
composable. The GPU operations naturally map to asyncio tasks on separate CUDA streams.

### 3. Parse config.json, don't write per-model code

fastllm has ~76K lines of model-specific C++ (`src/models/*.cpp`). The Python engine
should have zero model-specific code. Parse the HuggingFace config dict and
reconstruct the compute graph from standard layer types (LlamaAttention, QwenMLP,
DeepseekMoE, etc.). This is what vLLM does — but vLLM still needs per-model Python
files for architecture variants. We can go further by using only the config keys.

### 4. Support speculative decoding from day one

The draft model is just another instance of the engine with a tiny model (0.5B params).
The verification step is a single forward pass of the target model. This 2-4× speedup
is pure software and should be built into the scheduler from the start.

---

## Predicted Performance

Single-stream decode on DeepSeek V4 671B, 1×RTX 4090 + 256 GB DDR5 + dual EPYC:

| Optimization | tok/s |
|--------------|-------|
| Baseline (naive all-CPU) | ~2 |
| + Hybrid GPU/CPU MoE | ~12 |
| + Expert LRU cache (80% hit rate) | ~18 |
| + Cross-layer prefetch | ~22 |
| + Pinned weight + async DMA merge | ~25 |
| + CUDA Graph for attention | ~28 |
| + Speculative decoding (draft model) | ~45-60 |

On a 4090 with 24 GB (all experts at INT4 fit in VRAM): ~40-60 tok/s.
