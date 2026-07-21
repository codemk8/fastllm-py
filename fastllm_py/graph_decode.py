"""CUDA-graph-accelerated single-token decode for dense (non-MLA) INT4 models.

Decode is host-dispatch-bound: each token issues ~hundreds of tiny kernel
launches whose Python/driver overhead dwarfs the GPU work. A CUDA graph
captures that per-token step once and replays it as a single launch.

A CUDA graph is *per-device*, so a model whose layers are split across GPUs is
captured as **one graph per contiguous device-segment**, chained by a small
host-hopped copy of the boundary hidden state (16 KB — the 4090s have no P2P).
N=1 GPU is just the single-segment case. Per token this is N graph launches +
(N-1) tiny copies instead of ~hundreds of kernel launches.

Design invariants (per segment):
  * one capturable (non-blocking) stream on the segment's device;
  * static shapes — KV is preallocated to ``max_len``; attention is a
    flash-decode RawKernel that loops only over the valid ``[0, pos]`` keys
    (``valid_len = pos+1`` read from the device pos buffer), so its cost is
    O(valid_len), not O(max_len);
  * static addresses — fixed input/pos/output buffers updated in place;
  * no host sync inside the step — the new K/V row is written by a RawKernel at
    a device-resident position; RoPE is a gather from a precomputed table.

Correctness notes: linear layers must be Marlin INT4 (cuBLAS is rejected during
cupy stream capture); attention is done as broadcast-multiply + reductions (no
matmul); lm_head (fp16/cuBLAS) runs outside the graph. Input writes MUST be
issued on the capture stream (a default-stream write race was a subtle bug —
found via compute-sanitizer). Each segment uses a dedicated capture mem-pool
and per-call Marlin workspaces. `verify()` checks bit-exactness vs eager and
`generate()` falls back to eager if a model ever diverges.

Scope: dense non-MLA models (Qwen3, Llama/DeepSeek-LLM, Qwen2), any GPU count.
MLA/MoE fall back to eager decode.
"""
from __future__ import annotations

import os

import numpy as np

from .kernels.ops import apply_rope, rmsnorm, swiglu
from .model import matmul_w

# --------------------------------------------------------------------------
# Activation-dtype contract (mixed precision).
#
# The residual/activation stream is NOT guaranteed to be a single dtype: some
# ops accumulate in higher precision (the MoE branch and attention produce fp32)
# and the residual absorbs that. Every kernel that CONSUMES an activation must
# therefore coerce its input to the *compute* dtype it was compiled for — do it
# through GraphDecoder._act_in() (never assume the incoming dtype). Custom
# RawKernels are compiled per compute-dtype via the ctype string from
# _act_ctype(); to add a new activation/compute precision (bf16, fp8, ...) add
# one entry to _ACT_CTYPE and make sure the kernels support it. Set
# FASTLLM_CHECK_DTYPE=1 to assert, at every kernel boundary, that activations
# only ever flow in a registered dtype (catches a new op that forgets to cast).
# --------------------------------------------------------------------------
_ACT_CTYPE: dict = {"float16": "__half", "float32": "float"}
_CHECK_DTYPE = os.environ.get("FASTLLM_CHECK_DTYPE") == "1"


def _act_ctype(dtype) -> str:
    """CUDA C scalar type for an activation/compute dtype. Raises (rather than
    silently mis-reading memory) on an unregistered dtype — the single place to
    extend for a new precision."""
    name = np.dtype(dtype).name
    if name not in _ACT_CTYPE:
        raise TypeError(
            f"activation dtype {name!r} has no registered kernel ctype; add it "
            f"to _ACT_CTYPE (and ensure the kernels compile for it)")
    return _ACT_CTYPE[name]


def apply_penalties(logits, counts, repetition_penalty: float = 1.0,
                    frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0) -> np.ndarray:
    """Return logits adjusted for previously generated tokens (a dict
    token_id -> count). repetition_penalty is HF-style (divide positive logits,
    multiply negative ones); frequency/presence are OpenAI-style (subtract
    freq*count + presence*[seen]). No-op (returns input) when nothing is set or
    counts is empty."""
    if not counts or (repetition_penalty == 1.0 and frequency_penalty == 0.0
                      and presence_penalty == 0.0):
        return logits
    lg = np.array(logits, dtype=np.float64, copy=True)
    ids = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    cnt = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
    if repetition_penalty != 1.0:
        sub = lg[ids]
        lg[ids] = np.where(sub > 0, sub / repetition_penalty,
                           sub * repetition_penalty)
    if frequency_penalty != 0.0 or presence_penalty != 0.0:
        lg[ids] -= frequency_penalty * cnt + presence_penalty
    return lg


def logits_to_probs(logits, temperature: float, top_p: float = 1.0,
                    top_k: int = 0, min_p: float = 0.0) -> np.ndarray:
    """(vocab,) logits -> normalized probability vector after temperature scale,
    optional top-k, min-p, and nucleus (top-p) truncation. temperature must be
    > 0. This is the sampling distribution both the plain sampler and
    speculative sampling draw from (so their transforms stay identical)."""
    lg = np.asarray(logits, dtype=np.float64) / temperature
    lg -= lg.max()
    probs = np.exp(lg)
    probs /= probs.sum()
    if top_k and 0 < top_k < probs.size:
        drop = np.argpartition(probs, -top_k)[:-top_k]
        probs[drop] = 0.0
        probs /= probs.sum()
    if min_p > 0.0:                    # keep tokens >= min_p * peak probability
        probs[probs < min_p * probs.max()] = 0.0
        probs /= probs.sum()
    if top_p < 1.0:
        order = np.argsort(-probs)
        csum = np.cumsum(probs[order])
        cutoff = order[csum > top_p]
        if len(cutoff) > 1:            # keep the first token to cross top_p
            probs[cutoff[1:]] = 0.0
            probs /= probs.sum()
    return probs


def sample_logits(logits, temperature: float = 0.0, top_p: float = 1.0,
                  top_k: int = 0, rng=None, min_p: float = 0.0) -> int:
    """Pick a token id from a (vocab,) numpy logit vector. temperature<=0 is
    greedy argmax; otherwise temperature-scale -> optional top-k -> min-p ->
    nucleus (top-p) -> categorical sample. rng is an optional np.random.Generator
    for reproducibility (falls back to the global RNG). Any generation penalties
    should already be applied to `logits` (see apply_penalties)."""
    if temperature <= 0.0:
        return int(np.argmax(logits))
    probs = logits_to_probs(logits, temperature, top_p, top_k, min_p)
    return int((rng or np.random).choice(probs.size, p=probs))


def route_gpu(logits, top_k, scoring="softmax", bias=None, n_group=0,
              topk_group=0, norm=False, scale=1.0):
    """On-GPU expert routing using ONLY capturable ops (exp/sort/where/sum —
    no cuBLAS, no host sync), so it runs inside a CUDA graph. Returns a
    (T, E) weight matrix, 0 for non-selected experts. Matches the CPU
    expert_router.route_topk selection (incl. DeepSeek V3/V4 group-limited
    routing) to float32 precision. Foundation for dense-over-experts graph
    MoE decode (compute all E experts, weight-mask to the selected top-k)."""
    import cupy as cp

    T, E = logits.shape
    if scoring == "sigmoid":
        probs = 1.0 / (1.0 + cp.exp(-logits))
        select = probs + (bias if bias is not None else 0.0)
    else:
        e = cp.exp(logits - logits.max(1, keepdims=True))
        probs = e / e.sum(1, keepdims=True)
        select = probs
    if n_group > 1 and 0 < topk_group < n_group:
        gsz = E // n_group
        gs = select.reshape(T, n_group, gsz)
        grp = cp.sort(gs, axis=-1)[:, :, -min(2, gsz):].sum(-1)
        gthr = cp.sort(grp, axis=-1)[:, -topk_group][:, None]
        select = cp.where(cp.repeat(grp >= gthr, gsz, axis=1), select, -cp.inf)
    kth = cp.sort(select, axis=-1)[:, -top_k][:, None]
    w = cp.where(select >= kth, probs, 0.0)
    if norm:
        w = w / (w.sum(1, keepdims=True) + 1e-20)
    return (w * scale).astype(cp.float32)


_DENSE_GEMV: dict = {}


def _dense_gemv_kernel(cp, ctype: str):
    """Dense row-major INT4 GEMV for M=1 decode: x staged in shared once, one
    warp per output row with a shuffle reduction, vectorized 4-byte INT4 loads.
    The M=1-efficient replacement for Marlin (a GEMM kernel) on the attention
    projections. ctype = fp16/fp32 for x and y; fp32 accumulation."""
    if ctype not in _DENSE_GEMV:
        src = r"""
        #include <cuda_fp16.h>
        __device__ __forceinline__ float wrd(
                const unsigned char* qw, const __half* sc, const __half* ze,
                int row, const CT* vec, int in_f, int gs, int lane) {
            const unsigned char* wr = qw + (long long)row * (in_f / 2);
            const __half* s = sc + (long long)row * (in_f / gs);
            const __half* z = ze + (long long)row * (in_f / gs);
            float acc = 0.0f;
            for (int p = lane * 8; p < in_f; p += 256) {
                int gi = p / gs; float sv = __half2float(s[gi]), zv = __half2float(z[gi]);
                unsigned int pk = *reinterpret_cast<const unsigned int*>(wr + (p >> 1));
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    unsigned char b = (pk >> (j * 8)) & 0xFF;
                    acc += ((float)(b & 0xF) - zv) * sv * (float)vec[p + 2 * j];
                    acc += ((float)(b >> 4) - zv) * sv * (float)vec[p + 2 * j + 1];
                }
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
            return acc;
        }
        extern "C" __global__ void dgemv(
                const CT* __restrict__ x, const unsigned char* __restrict__ qw,
                const __half* __restrict__ sc, const __half* __restrict__ ze,
                CT* __restrict__ y, int in_f, int out_f, int gs, int rpb) {
            extern __shared__ CT xs[];   // stage x in activation dtype: half the
            for (int i = threadIdx.x; i < in_f; i += blockDim.x)  // smem of fp32
                xs[i] = x[i];            // -> more blocks/SM (occupancy win)
            __syncthreads();
            int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, nw = blockDim.x >> 5;
            int tile = blockIdx.x * rpb;
            for (int r = tile + wid; r < tile + rpb && r < out_f; r += nw) {
                float v = wrd(qw, sc, ze, r, xs, in_f, gs, lane);
                if (lane == 0) y[r] = (CT)v;
            }
        }
        """.replace("CT", ctype)
        _DENSE_GEMV[ctype] = cp.RawKernel(src, "dgemv")
    return _DENSE_GEMV[ctype]


def graph_capable(model) -> bool:
    """True if GraphDecoder supports this model: INT4 (Marlin dict) linears,
    non-MLA. MoE is supported on a single GPU when experts are INT4-resident
    (moe_device={"cuda":1} + gpu_expert_quant="int4") — the fused custom-kernel
    MoE path. Dense supports any GPU count."""
    cfg = model.cfg
    if cfg.is_mla:
        return False
    if any(not l.device.startswith("cuda") for l in model.layers):
        return False
    if not isinstance(model.layers[0].w.get("self_attn.q_proj.weight"), dict):
        return False
    if cfg.is_moe:
        if len({l.device for l in model.layers}) != 1:
            return False  # fused MoE kernel is single-GPU
        ml = next((l.moe for l in model.layers if l.moe is not None), None)
        return ml is not None and ml.gpu_payloads is not None
    return True


_ROUTE_TOPK = None


def _route_topk_kernel(cp):
    """Fused softmax + top-K + optional renorm routing in one block (was ~13
    elementwise launches + a full radix sort per MoE layer -> ~6x faster,
    bit-exact). Writes the (E,) routing-weight vector: 0 for non-top-K experts,
    else softmax prob (renormalized over the kept K if norm) * scale. Top-K by
    count-greater (ties -> keep, matching where(probs >= kth))."""
    global _ROUTE_TOPK
    if _ROUTE_TOPK is None:
        _ROUTE_TOPK = cp.RawKernel(r"""
        extern "C" __global__ void route_topk(
                const float* __restrict__ logits, float* __restrict__ rw,
                int E, int K, int norm, float scale) {
            int tid = threadIdx.x, nt = blockDim.x;
            extern __shared__ float sh[];          // pr[E] + red[nt]
            float* pr = sh; float* red = sh + E;
            float mx = -1e30f;
            for (int i = tid; i < E; i += nt) mx = fmaxf(mx, logits[i]);
            red[tid] = mx; __syncthreads();
            for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid+s]); __syncthreads(); }
            float m = red[0]; __syncthreads();
            float z = 0.0f;
            for (int i = tid; i < E; i += nt) { float p = __expf(logits[i] - m); pr[i] = p; z += p; }
            red[tid] = z; __syncthreads();
            for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }
            float Z = red[0]; __syncthreads();
            float ks = 0.0f;
            for (int i = tid; i < E; i += nt) {
                float pi = pr[i]; int c = 0;
                for (int j = 0; j < E; j++) c += (pr[j] > pi);
                if (c < K) ks += pi;
            }
            red[tid] = ks; __syncthreads();
            for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }
            float denom = norm ? (red[0] + 1e-20f) : Z;
            __syncthreads();
            for (int i = tid; i < E; i += nt) {
                float pi = pr[i]; int c = 0;
                for (int j = 0; j < E; j++) c += (pr[j] > pi);
                rw[i] = (c < K) ? (pi / denom) * scale : 0.0f;
            }
        }
        """, "route_topk")
    return _ROUTE_TOPK


_ROPE_K: dict = {}


def _rope_kernel(cp, ctype: str):
    """Fused RoPE (HF rotate_half) in one kernel — replaces apply_rope's
    astype(fp32) + 4 multiplies + 2 adds + concatenate + astype-back (~8
    elementwise launches, a big slice of the decode 'soup'; nsys cupy_multiply
    was entirely rope). One thread per (head, dim). cos/sin: (D/2,) fp32."""
    if ctype not in _ROPE_K:
        _ROPE_K[ctype] = cp.RawKernel(rf"""
        #include <cuda_fp16.h>
        extern "C" __global__ void rope(
                const {ctype}* __restrict__ qin, const float* __restrict__ cs,
                const float* __restrict__ sn, {ctype}* __restrict__ qout,
                int nheads, int D) {{
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= nheads * D) return;
            int d = idx % D, half = D >> 1, h = idx / D;
            if (d < half) {{
                float q1 = static_cast<float>(qin[idx]);
                float q2 = static_cast<float>(qin[h * D + d + half]);
                qout[idx] = ({ctype})(q1 * cs[d] - q2 * sn[d]);
            }} else {{
                int e = d - half;
                float q2 = static_cast<float>(qin[idx]);
                float q1 = static_cast<float>(qin[h * D + e]);
                qout[idx] = ({ctype})(q2 * cs[e] + q1 * sn[e]);
            }}
        }}""", "rope")
    return _ROPE_K[ctype]


_ATTN_DECODE: dict = {}


def _attn_decode_kernel(cp, ctype: str):
    """Flash-decode attention RawKernel: one block per query head (GQA-aware,
    kv_head = h/(H/KVH)), online softmax in fp32. Cost is O(valid_len) (loops
    only over VALID keys [0, pos] from the device pos buffer).

    Two-phase, no per-key block reduction (the old kernel did a D-way shared
    reduction *per key* — ~7 __syncthreads/key — and ran at ~3% of BW; ~4x
    faster here): (1) each thread scores a subset of keys (full D dot-product),
    (2) block softmax over the VALID keys, (3) each thread accumulates one output
    dim over all keys. `scores` is staged in shared, so shared_mem must hold
    max_len floats — fine to ~8-12K context; longer needs a split-KV variant.
    q/out are fp32; kc/vc are `ctype`. Block is `threads` (>= D), not tied to D."""
    key = ctype
    if key not in _ATTN_DECODE:
        _ATTN_DECODE[key] = cp.RawKernel(rf"""
        #include <cuda_fp16.h>
        extern "C" __global__ void attn_decode_{ctype.replace(' ', '_')}(
                const {ctype}* __restrict__ q,     // (H, D) activation dtype
                const {ctype}* __restrict__ kc,    // (max_len, KVH, D)
                const {ctype}* __restrict__ vc,    // (max_len, KVH, D)
                {ctype}* __restrict__ out,         // (H, D) activation dtype
                const int* __restrict__ pos,       // device scalar; valid = pos[0]+1
                int H, int KVH, int D, float scale) {{
            int h = blockIdx.x, tid = threadIdx.x, BLK = blockDim.x;
            int rep = H / KVH, kvh = h / rep, VL = pos[0] + 1;
            extern __shared__ float sh[];
            float* sq = sh;                        // D
            float* scores = sh + D;                // VL (<= max_len)
            float* red = scores + VL;              // BLK
            for (int i = tid; i < D; i += BLK) sq[i] = (float)q[h * D + i];
            __syncthreads();
            // (1) scores[j] = dot(q, k_j) * scale
            for (int j = tid; j < VL; j += BLK) {{
                long long base = ((long long)j * KVH + kvh) * D;
                float acc = 0.0f;
                for (int d = 0; d < D; d++) acc += sq[d] * (float)kc[base + d];
                scores[j] = acc * scale;
            }}
            __syncthreads();
            // (2) softmax over VALID keys: block max, then exp + block sum
            float mx = -1e30f;
            for (int j = tid; j < VL; j += BLK) mx = fmaxf(mx, scores[j]);
            red[tid] = mx; __syncthreads();
            for (int s = BLK >> 1; s > 0; s >>= 1) {{ if (tid < s) red[tid] = fmaxf(red[tid], red[tid+s]); __syncthreads(); }}
            float m = red[0]; __syncthreads();
            float sm = 0.0f;
            for (int j = tid; j < VL; j += BLK) {{ float e = __expf(scores[j] - m); scores[j] = e; sm += e; }}
            red[tid] = sm; __syncthreads();
            for (int s = BLK >> 1; s > 0; s >>= 1) {{ if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }}
            float l = red[0]; __syncthreads();
            // (3) out[d] = sum_j p_j * v_j[d] / l  (one dim per thread)
            for (int d = tid; d < D; d += BLK) {{
                float acc = 0.0f;
                for (int j = 0; j < VL; j++) acc += scores[j] * (float)vc[((long long)j * KVH + kvh) * D + d];
                out[h * D + d] = ({ctype})(acc / l);
            }}
        }}""", f"attn_decode_{ctype.replace(' ', '_')}")
    return _ATTN_DECODE[key]


_WRITE_KV: dict = {}


def _write_kv_kernel(cp, ctype: str):
    """RawKernel: copy the current token's K/V rows into the cache at a
    device-resident position (capturable — index comes from a device buffer).
    Keyed by element C type so fp32 and fp16 caches both work."""
    if ctype not in _WRITE_KV:
        _WRITE_KV[ctype] = cp.RawKernel(rf"""
        #include <cuda_fp16.h>
        extern "C" __global__ void write_kv_{ctype.replace(' ', '_')}(
                {ctype}* kcache, {ctype}* vcache,
                const {ctype}* knew, const {ctype}* vnew,
                const int* pos, int row_size) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < row_size) {{
                long long p = (long long)pos[0] * row_size + i;
                kcache[p] = knew[i];
                vcache[p] = vnew[i];
            }}
        }}""", f"write_kv_{ctype.replace(' ', '_')}")
    return _WRITE_KV[ctype]


class _Segment:
    """A contiguous run of decoder layers on one GPU, plus its capture state."""

    def __init__(self, decoder, dev_id, layers, layer_offset, is_last):
        cp = decoder.cp
        cfg = decoder.cfg
        self.dev_id = dev_id
        self.layers = layers
        self.layer_offset = layer_offset  # global index of layers[0]
        self.is_last = is_last
        self.graph = None
        self.ws_list = []   # per-Marlin-call-site workspaces (see _mm)
        self.ws_i = 0
        self._pool = None
        self._decoder = decoder
        cp = decoder.cp
        cfg = decoder.cfg
        hidden, dt = cfg.hidden_dim, decoder.dtype
        with cp.cuda.Device(dev_id):
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.x_in = cp.zeros((1, hidden), dtype=dt)      # segment input hidden
            self.pos_idx = cp.zeros((1,), dtype=cp.int32)
            self.hidden_out = cp.zeros((hidden,), dtype=dt)  # segment output hidden
        self.alloc(decoder.max_len)

    def alloc(self, max_len: int):
        """(Re)allocate the max_len-sized buffers. Attention is O(max_len) per
        token (it scans the whole bias-masked buffer), so sizing max_len to the
        actual sequence — not a fixed large default — is a big win for wide
        models. Called on construction and by GraphDecoder.resize()."""
        cp = self._decoder.cp
        cfg = self._decoder.cfg
        H, KVH, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        dt = self._decoder.dtype
        self.max_len = max_len
        self.graph = None          # invalidate any prior capture
        self.ws_list, self.ws_i = [], 0
        with cp.cuda.Device(self.dev_id):
            self.k_cache = [cp.zeros((max_len, KVH, D), dtype=dt) for _ in self.layers]
            self.v_cache = [cp.zeros((max_len, KVH, D), dtype=dt) for _ in self.layers]
            cos, sin = self._decoder.model._rope_cache(cp.arange(max_len), D, cp)
            self.cos_tab, self.sin_tab = cos, sin


class GraphDecoder:
    def __init__(self, model, max_len: int = 256):
        import cupy as cp

        cfg = model.cfg
        if cfg.is_mla:
            raise ValueError("GraphDecoder does not support MLA models")
        if not isinstance(model.layers[0].w.get("self_attn.q_proj.weight"), dict):
            raise ValueError("GraphDecoder requires an INT4 model "
                             "(Model.load(..., linear_quant='int4'))")
        if cfg.is_moe and not graph_capable(model):
            raise ValueError("GraphDecoder MoE needs INT4-resident experts "
                             "(moe_device={'cuda':1}, gpu_expert_quant='int4') on 1 GPU")

        self.cp = cp
        self.model = model
        self.cfg = cfg
        self.max_len = max_len
        self.dtype = cp.float32 if model.dtype == "float32" else cp.float16
        # compute dtype for custom RawKernels; activations are coerced to it via
        # _act_in() at every kernel boundary (see the module-level contract).
        self.ctype = _act_ctype(self.dtype)

        # group consecutive layers by device into segments
        self.segments = []
        cur_dev, start = model.layers[0].device, 0
        for i, layer in enumerate(model.layers + [None]):
            dev = layer.device if layer is not None else None
            if dev != cur_dev:
                seg_layers = model.layers[start:i]
                dev_id = int(cur_dev.split(":")[1]) if ":" in cur_dev else 0
                self.segments.append(
                    _Segment(self, dev_id, seg_layers, start,
                             is_last=(i == len(model.layers))))
                cur_dev, start = dev, i

        self.embed_dev = self.segments[0].dev_id
        self.head_dev = self.segments[-1].dev_id
        with cp.cuda.Device(self.head_dev):
            self.logits = cp.zeros((model.lm_head.shape[0],), dtype=self.dtype)
        self._captured = False
        self.graph_fellback = False
        if cfg.is_moe:
            self._prepare_moe()
        self._prepare_dense_linears()

    def _prepare_dense_linears(self):
        """Re-quantize the attention projections (q/k/v/o) to row-major INT4 and
        stash them per layer as `layer._rmw`, so the capturable decode path can
        use the lean warp-per-row GEMV instead of Marlin. Marlin is a GEMM kernel
        tiled for M>=16; at decode M=1 it runs at ~1/3 the row-major GEMV's speed
        (nsys: q/k/v/o projections were 27% of the token). Falls back silently to
        Marlin for any layer whose fp16 weights can't be re-read."""
        cp = self.cp
        import os
        from .kernels.moe_int4 import quantize_int4_rowmajor

        store = getattr(self.model, "store", None)
        if store is None or os.environ.get("FASTLLM_NO_DGEMV") == "1":  # default on; escape hatch reverts projections to Marlin
            return
        prefix = "model." if "model.layers.0.self_attn.q_proj.weight" in store else ""
        names = ("self_attn.q_proj", "self_attn.k_proj",
                 "self_attn.v_proj", "self_attn.o_proj")
        for seg in self.segments:
            with cp.cuda.Device(seg.dev_id):
                for layer in seg.layers:
                    rmw = {}
                    try:
                        for n in names:
                            key = f"{prefix}layers.{layer.idx}.{n}.weight"
                            w = store.get_f32(key)          # (out, in) host fp32
                            rmw[n] = quantize_int4_rowmajor(w, 128, xp=cp)
                    except Exception:
                        rmw = None                          # keep Marlin for this layer
                    layer._rmw = rmw
                # persistent output buffers per projection (NO alloc during graph
                # capture — one buffer per name, reused across this segment's
                # layers; k/v need distinct buffers as they're live together).
                seg._proj_bufs = {}
                ref = next((l._rmw for l in seg.layers if getattr(l, "_rmw", None)), None)
                if ref is not None:
                    for n, p in ref.items():
                        seg._proj_bufs[n] = cp.empty((p["shape"][0],), dtype=self.dtype)

    def _prepare_moe(self):
        """Build per-MoE-layer graph data: fp16 gate weights + row-major INT4
        stacked experts (routed + shared), plus reusable device scratch. All on
        the single GPU. Enables the capturable fused MoE decode branch."""
        cp = self.cp
        from .kernels import moe_int4

        cfg = self.cfg
        dev = self.segments[0].dev_id
        E = cfg.num_experts
        first = next(l.moe for l in self.model.layers if l.moe is not None)
        inter = first._materialize_cpu(0)["gate"].shape[0]
        sinter = first.shared["gate"].shape[0] if first.shared is not None else 0
        self._moe = {"E": E, "inter": inter, "sinter": sinter,
                     "has_shared": first.shared is not None,
                     "has_sgate": first.shared_gate is not None}
        with cp.cuda.Device(dev):
            self._moe_lbuf = cp.empty((E,), dtype=cp.float32)
            self._moe_ibuf = cp.empty((E, inter), dtype=cp.float32)
            self._moe_rw = cp.empty((E,), dtype=cp.float32)
            self._moe_out = cp.empty((cfg.hidden_dim,), dtype=cp.float32)
            self._moe_srw = cp.ones((1,), dtype=cp.float32)
            self._moe_sout = cp.empty((cfg.hidden_dim,), dtype=cp.float32)
            self._moe_sibuf = cp.empty((1, max(sinter, 1)), dtype=cp.float32)
            # Startup cache for the row-major stacked experts: building them
            # re-reads + re-quantizes every expert (minutes for a 30B). Cache the
            # finished int4 payloads per layer; warm starts stream them straight
            # from disk to GPU. (Cache is written by the same first run that
            # builds the marlin expert cache, so the fp16 experts are only ever
            # read once.)
            from pathlib import Path

            store = getattr(self.model, "store", None)
            rdir = (Path(store.model_path) / ".rowmajor_cache") if store else None

            def _save_stacked(path, st):
                out = {}
                for k, v in st.items():
                    if k == "group":
                        out[k] = np.int64(v)
                    elif k.endswith(".shape"):
                        out[k] = np.asarray(v, dtype=np.int64)
                    else:
                        out[k] = cp.asnumpy(v)
                path.parent.mkdir(exist_ok=True)
                tmp = path.with_name(path.name + ".tmp.npz")
                np.savez(tmp, **out)
                tmp.rename(path)

            def _load_stacked(path):
                z = np.load(path)
                st = {}
                for k in z.files:
                    if k == "group":
                        st[k] = int(z[k])
                    elif k.endswith(".shape"):
                        st[k] = tuple(int(x) for x in z[k])
                    else:
                        st[k] = cp.asarray(z[k])
                return st

            for layer in self.model.layers:
                ml = getattr(layer, "moe", None)
                if ml is None:
                    continue
                rc = (rdir / f"experts.L{layer.idx}.npz") if rdir else None
                sc = (rdir / f"shared.L{layer.idx}.npz") if rdir else None
                if rc is not None and rc.exists():
                    stacked = _load_stacked(rc)
                    shared = (_load_stacked(sc)
                              if ml.shared is not None and sc.exists() else
                              (moe_int4.build_stacked_experts([ml.shared], 128, xp=cp)
                               if ml.shared is not None else None))
                else:
                    ews = [ml._materialize_cpu(e) for e in range(E)]
                    stacked = moe_int4.build_stacked_experts(ews, 128, xp=cp)
                    shared = (moe_int4.build_stacked_experts([ml.shared], 128, xp=cp)
                              if ml.shared is not None else None)
                    if rc is not None:
                        _save_stacked(rc, stacked)
                        if shared is not None:
                            _save_stacked(sc, shared)
                layer._gmoe = {
                    "gate_w": ml.gate_weight.astype(cp.float16),
                    "stacked": stacked,
                    "shared": shared,
                    "shared_gate_w": (ml.shared_gate.astype(cp.float16)
                                      if ml.shared_gate is not None else None),
                }

    def _moe_branch(self, layer, h):
        """Capturable fused MoE FFN for one decode token. h: (1, hidden).
        Returns (hidden,) fp32: routed experts + shared expert."""
        cp = self.cp
        cfg = self.cfg
        from .kernels import moe_int4

        gm = layer._gmoe
        M = self._moe
        xr = h[0]
        # gate (custom matvec, no cuBLAS) + fused softmax/top-k/renorm routing
        moe_int4.gate_matvec(xr, gm["gate_w"], M["E"], cfg.hidden_dim, out=self._moe_lbuf)
        E = M["E"]
        _route_topk_kernel(cp)((1,), (128,),
            (self._moe_lbuf, self._moe_rw, np.int32(E),
             np.int32(cfg.num_experts_per_tok),
             np.int32(1 if cfg.norm_topk_prob else 0),
             np.float32(cfg.routed_scaling_factor)),
            shared_mem=(E + 128) * 4)
        moe_int4.fused_moe_weighted(xr, gm["stacked"], self._moe_rw, M["E"],
                                    cfg.hidden_dim, M["inter"], out=self._moe_out,
                                    inter_buf=self._moe_ibuf)
        if gm["shared"] is not None:
            moe_int4.fused_moe_weighted(xr, gm["shared"], self._moe_srw, 1,
                                        cfg.hidden_dim, M["sinter"], out=self._moe_sout,
                                        inter_buf=self._moe_sibuf)
            if gm["shared_gate_w"] is not None:
                sg = moe_int4.gate_matvec(xr, gm["shared_gate_w"], 1, cfg.hidden_dim)
                self._moe_out += self._moe_sout * (1.0 / (1.0 + cp.exp(-sg[0])))
            else:
                self._moe_out += self._moe_sout
        return self._moe_out

    # ------------------------------------------------------------ dtype contract
    def _act_in(self, t):
        """Coerce an activation to the compute dtype the custom kernels expect.
        The residual stream may be fp32 (the MoE branch / attention accumulate in
        fp32), so every kernel that reads an activation MUST route it through
        here — a kernel compiled for __half would otherwise reinterpret fp32
        bytes as fp16 pairs and produce NaNs. No-op when already the right dtype.
        With FASTLLM_CHECK_DTYPE=1, asserts the incoming dtype is registered."""
        if _CHECK_DTYPE and np.dtype(t.dtype).name not in _ACT_CTYPE:
            raise TypeError(f"activation dtype {t.dtype} not registered (see "
                            "_ACT_CTYPE); a new op is emitting an unhandled dtype")
        return t if t.dtype == self.dtype else t.astype(self.dtype)

    def _rope(self, x, cos, sin, nheads):
        """Fused RoPE over x (…, nheads, D). cos/sin are (1, D/2) fp32 (the
        precomputed table row for this position). Returns a new array."""
        cp = self.cp
        D = self.cfg.head_dim
        ker = _rope_kernel(cp, self.ctype)
        out = cp.empty_like(x)
        n = nheads * D
        ker(((n + 255) // 256,), (256,),
            (x.reshape(-1), cos.ravel(), sin.ravel(), out.reshape(-1),
             np.int32(nheads), np.int32(D)))
        return out

    # ------------------------------------------------------------ per segment
    def _dgemv(self, seg, inp, payload, buf):
        """Dense row-major INT4 GEMV (warp-per-row, x staged in shared, shuffle
        reduce) — the M=1-efficient replacement for Marlin on the projections.
        inp: (1, in) any registered activation dtype (coerced to self.dtype).
        Writes into the persistent `buf` (no alloc during capture) -> (1, out)."""
        cp = self.cp
        k = _dense_gemv_kernel(cp, self.ctype)
        out_f, in_f = payload["shape"]
        gs = payload["group"]
        x = self._act_in(inp).reshape(-1)
        # rpb=16/th=512 + fp16-staged x won a graph-timed sweep (1.1-1.2x over
        # rpb8/th256/fp32-shared) — the smaller smem lifts blocks/SM.
        rpb = 16
        k(((out_f + rpb - 1) // rpb,), (512,),
          (x, payload["qweight"], payload["scales"], payload["zeros"], buf,
           np.int32(in_f), np.int32(out_f), np.int32(gs), np.int32(rpb)),
          shared_mem=in_f * x.dtype.itemsize)
        return buf.reshape(1, out_f)

    def _mm(self, seg, inp, w):
        """Marlin INT4 GEMV on the segment's stream with a per-call-site
        workspace (a shared workspace corrupts a replayed graph)."""
        from .kernels.marlin import gemm_fast, make_workspace

        size_n = w["scales"].shape[1]
        i = seg.ws_i
        if i >= len(seg.ws_list):
            seg.ws_list.append(make_workspace(size_n, self.cp))
        seg.ws_i += 1
        a = inp if inp.dtype == self.cp.float16 else inp.astype(self.cp.float16)
        return gemm_fast(a, w["qweight"], w["scales"], w["zeros"],
                         size_n, inp.shape[1], stream=seg.stream,
                         workspace=seg.ws_list[i])

    def _segment_step(self, seg):
        """Issue this segment's layers on seg.stream, reading seg.x_in /
        pos_idx and writing seg.hidden_out. Static shapes, no host sync."""
        cp = self.cp
        cfg = self.cfg
        H, KVH, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        assert (D & (D - 1)) == 0, "flash-decode kernel needs power-of-two head_dim"
        scale = D ** -0.5
        ctype = self.ctype
        write = _write_kv_kernel(cp, ctype)
        attn = _attn_decode_kernel(cp, ctype)
        row = KVH * D
        blocks = (row + 127) // 128
        cos = seg.cos_tab[seg.pos_idx]
        sin = seg.sin_tab[seg.pos_idx]
        x = seg.x_in
        seg.ws_i = 0

        def lin(layer, name, inp):
            rmw = getattr(layer, "_rmw", None)
            if rmw is not None and name in rmw:      # row-major GEMV (M=1 fast)
                out = self._dgemv(seg, inp, rmw[name], seg._proj_bufs[name])
            else:
                out = self._mm(seg, inp, layer.w[f"{name}.weight"])
            if layer.has(f"{name}.bias"):
                out = out + layer.w[f"{name}.bias"]
            return out

        for li, layer in enumerate(seg.layers):
            h = rmsnorm(x, layer.w["input_layernorm.weight"], cfg.norm_eps)
            q = lin(layer, "self_attn.q_proj", h).reshape(1, H, D)
            k = lin(layer, "self_attn.k_proj", h).reshape(1, KVH, D)
            v = lin(layer, "self_attn.v_proj", h).reshape(1, KVH, D)
            if layer.has("self_attn.q_norm.weight"):
                q = rmsnorm(q, layer.w["self_attn.q_norm.weight"], cfg.norm_eps)
                k = rmsnorm(k, layer.w["self_attn.k_norm.weight"], cfg.norm_eps)
            q = self._rope(q, cos, sin, H)
            k = self._rope(k, cos, sin, KVH)

            kc, vc = seg.k_cache[li], seg.v_cache[li]
            # KV cache + write kernel are compute-dtype; coerce k/v (activations)
            # through the dtype contract, same as dgemv.
            write((blocks,), (128,),
                  (kc, vc, cp.ascontiguousarray(self._act_in(k)),
                   cp.ascontiguousarray(self._act_in(v)),
                   seg.pos_idx, np.int32(row)))
            # flash-decode attention: O(valid_len), not O(max_len). Two-phase
            # kernel — one block per head, `_ABLK` threads; shared holds the
            # query (D), the per-key scores (up to max_len), and a reduction
            # scratch (_ABLK).
            # q is already the activation dtype (fp16) from _rope; the kernel
            # accumulates in fp32 internally, so no up/down-convert is needed —
            # feed q and receive ctx both in the activation dtype.
            qh = cp.ascontiguousarray(q.reshape(H, D))
            ctx = cp.empty((H, D), dtype=self.dtype)
            _ABLK = 128
            attn((H,), (_ABLK,), (qh, kc, vc, ctx, seg.pos_idx,
                                  np.int32(H), np.int32(KVH), np.int32(D),
                                  np.float32(scale)),
                 shared_mem=(D + self.max_len + _ABLK) * 4)
            ctx = ctx.reshape(1, H * D)
            x = x + lin(layer, "self_attn.o_proj", ctx)

            h = rmsnorm(x, layer.w["post_attention_layernorm.weight"], cfg.norm_eps)
            if getattr(layer, "_gmoe", None) is not None:
                x = x + self._moe_branch(layer, h)[None]
            else:
                g = self._mm(seg, h, layer.w["mlp.gate_proj.weight"])
                u = self._mm(seg, h, layer.w["mlp.up_proj.weight"])
                x = x + self._mm(seg, swiglu(g, u), layer.w["mlp.down_proj.weight"])

        if seg.is_last:  # final norm here; lm_head runs outside the graph
            seg.hidden_out[:] = rmsnorm(x, self.model.final_norm, cfg.norm_eps)[0]
        else:
            seg.hidden_out[:] = x[0]

    # ------------------------------------------------------------- prefill
    def prime(self, prompt_ids):
        """Eager prefill through the normal (possibly multi-GPU) model, then
        distribute the resulting KV into each segment's device buffers and mark
        valid positions. Returns last-token logits (numpy) + next position."""
        cp = self.cp
        logits, kvs = self.model.forward(np.asarray(prompt_ids))
        n = len(prompt_ids)
        if n > self.max_len - 1:
            raise ValueError(f"prompt {n} exceeds max_len {self.max_len}")
        for seg in self.segments:
            with cp.cuda.Device(seg.dev_id):
                for li in range(len(seg.layers)):
                    gi = seg.layer_offset + li
                    seg.k_cache[li][:n] = kvs[gi].k.astype(self.dtype)
                    seg.v_cache[li][:n] = kvs[gi].v.astype(self.dtype)
        last = logits[-1]
        return (cp.asnumpy(last) if isinstance(last, cp.ndarray) else last), n

    # ------------------------------------------------------------- capture
    def capture(self):
        cp = self.cp
        for seg in self.segments:
            with cp.cuda.Device(seg.dev_id):
                seg.pos_idx[0] = 0
                # dedicated pool: a captured graph bakes in its intermediates'
                # addresses; the shared pool would free+reuse them post-capture
                seg._pool = cp.cuda.MemoryPool()
                default_alloc = cp.get_default_memory_pool().malloc
                cp.cuda.set_allocator(seg._pool.malloc)
                try:
                    with seg.stream:
                        for _ in range(3):
                            self._segment_step(seg)
                    seg.stream.synchronize()
                    with seg.stream:
                        seg.stream.begin_capture()
                        self._segment_step(seg)
                    seg.graph = seg.stream.end_capture()
                finally:
                    cp.cuda.set_allocator(default_alloc)
                for c in seg.k_cache:
                    c.fill(0)
                for c in seg.v_cache:
                    c.fill(0)
                seg.stream.synchronize()
        self._captured = True

    def resize(self, need: int, headroom: int = 64):
        """Ensure the graph buffers hold at least `need` positions, sizing to a
        bucket (next power of two) so growth re-captures at most log(N) times.
        Attention cost is O(max_len)/token, so keeping max_len tight matters —
        especially for wide models (the 67B is 0.5x eager at max_len=2048 but
        1.3x at max_len~=need). Re-allocates + re-captures if grown."""
        target = 1
        while target < need + headroom:
            target *= 2
        if target == self.max_len and self._captured:
            return
        self.max_len = target
        for seg in self.segments:
            seg.alloc(target)
        self._captured = False
        self.capture()

    # ------------------------------------------------------------- stepping
    def _run(self, token_id, position, graph: bool):
        """One decode token across all segments; returns logits (numpy).

        Segments are pipelined with CUDA events + async cross-device copies of
        the boundary hidden state — a single host sync at the very end. Per-
        segment syncs would otherwise dominate for large models (where each
        segment's GPU work is big and the dispatch savings are small)."""
        cp = self.cp
        segs = self.segments
        prev_done = None  # event: prev segment's output hidden is ready
        for si, seg in enumerate(segs):
            with cp.cuda.Device(seg.dev_id), seg.stream:
                if si == 0:
                    seg.x_in[0] = self.model.embed[token_id].astype(self.dtype)
                else:
                    # boundary hidden: prev.hidden_out (dev si-1) -> x_in (dev si),
                    # async on seg.stream, ordered after prev_done (cross-device
                    # event wait). memcpyPeerAsync works without P2P (host-staged).
                    seg.stream.wait_event(prev_done)
                    prev = segs[si - 1]
                    nbytes = seg.x_in.size * seg.x_in.itemsize
                    cp.cuda.runtime.memcpyPeerAsync(
                        seg.x_in.data.ptr, seg.dev_id,
                        prev.hidden_out.data.ptr, prev.dev_id,
                        nbytes, seg.stream.ptr)
                seg.pos_idx[0] = position
                if graph:
                    seg.graph.launch(seg.stream)
                else:
                    self._segment_step(seg)
                prev_done = cp.cuda.Event()
                prev_done.record(seg.stream)
        last = segs[-1]
        with cp.cuda.Device(last.dev_id), last.stream:
            logits = matmul_w(last.hidden_out[None], self.model.lm_head)[0]
        last.stream.synchronize()  # the one and only host sync per token
        return cp.asnumpy(logits)

    def truncate(self, keep: int):
        """No-op with the flash-decode kernel: attention length is derived from
        the caller's `position` (valid_len = pos+1), not internal state, so
        rolling back is just passing a smaller position to the next step(). The
        stale KV rows past `keep` are simply never visited (and get overwritten).
        Kept for API compatibility (speculative decoding calls it)."""

    def step(self, token_id, position):
        return self._run(token_id, position, graph=True)

    def step_eager(self, token_id, position):
        return self._run(token_id, position, graph=False)

    # ------------------------------------------------------------- verify
    def verify(self, token_id, position, n=24, atol=0.0):
        """Bit-exact graph-vs-eager over n teacher-forced steps (on a throwaway
        KV snapshot). Returns True if they agree everywhere."""
        cp = self.cp

        def snap():
            s = []
            for seg in self.segments:
                with cp.cuda.Device(seg.dev_id):
                    s.append(([c.copy() for c in seg.k_cache],
                              [c.copy() for c in seg.v_cache]))
            return s

        def restore(s):
            for seg, (ks, vs) in zip(self.segments, s):
                with cp.cuda.Device(seg.dev_id), seg.stream:
                    for c, x in zip(seg.k_cache, ks):
                        c[...] = x
                    for c, x in zip(seg.v_cache, vs):
                        c[...] = x
                seg.stream.synchronize()

        orig, ref = snap(), snap()
        agree, tok, pos = True, token_id, position
        try:
            for _ in range(n):
                restore(ref)
                lg = self.step(tok, pos)
                restore(ref)
                le = self.step_eager(tok, pos)
                if float(np.abs(lg - le).max()) > atol:
                    agree = False
                    break
                ref = snap()
                tok = int(np.argmax(le))
                pos += 1
        finally:
            restore(orig)
        return agree

    # ------------------------------------------------------------- generate
    def generate(self, prompt_ids, max_new_tokens: int = 32, use_graph: bool = True,
                 verify: bool = True, stop_ids=None, temperature: float = 0.0,
                 top_p: float = 1.0, top_k: int = 0, seed=None, min_p: float = 0.0,
                 repetition_penalty: float = 1.0, frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0):
        if use_graph:
            self.resize(len(prompt_ids) + max_new_tokens)  # tight max_len (+capture)
        elif not self._captured:
            self.capture()
        first, pos = self.prime(prompt_ids)
        step = self.step if use_graph else self.step_eager
        self.graph_fellback = False
        # verify probes the graph teacher-forced on argmax (deterministic),
        # independent of the sampling mode used for the actual rollout.
        if use_graph and verify and not self.verify(int(np.argmax(first)), pos):
            step = self.step_eager
            self.graph_fellback = True
        rng = np.random.default_rng(seed) if temperature > 0.0 else None
        counts: dict = {}
        penalized = (repetition_penalty != 1.0 or frequency_penalty != 0.0
                     or presence_penalty != 0.0)

        def pick(lg):
            if penalized:
                lg = apply_penalties(lg, counts, repetition_penalty,
                                     frequency_penalty, presence_penalty)
            tok = sample_logits(lg, temperature, top_p, top_k, rng, min_p)
            counts[tok] = counts.get(tok, 0) + 1
            return tok

        out = [pick(first)]
        if stop_ids and out[-1] in stop_ids:
            return out
        for _ in range(max_new_tokens - 1):
            logits = step(out[-1], pos)
            pos += 1
            out.append(pick(logits))
            if stop_ids and out[-1] in stop_ids:
                break
        return out
