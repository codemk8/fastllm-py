"""Custom row-major INT4 group-quant GEMV — foundation for a fused selective
MoE kernel.

Marlin's tiled register layout is highly optimized but very hard to gather /
dequant from inside a custom kernel (needed for per-token selective experts).
This module uses a simple **row-major** INT4 group format we control, with the
SAME RTN group quantization as build_marlin_expert_payload, so results match the
marlin numeric path. Increment 2 of the fused-MoE plan (see
docs/next-optimizations.md): a single-expert GEMV; later increments fuse the
gate/up/down FFN and gather over the routed experts.

Format for a weight W of shape (out, in), group size gs along `in`:
  qweight : (out, in//2) uint8   — two int4 nibbles/byte, even col in low nibble
  scales  : (out, in//gs) fp16
  zeros   : (out, in//gs) fp16   — dequant(W[o,i]) = (q - zero) * scale
"""
from __future__ import annotations

import numpy as np

_GEMV = None


def quantize_int4_rowmajor(w, group_size: int = 128, xp=np) -> dict:
    """RTN group INT4, row-major, matching quantize_marlin_matrix's numerics."""
    w = xp.asarray(w)
    out_f, in_f = w.shape
    assert in_f % group_size == 0 and in_f % 2 == 0
    g = w.astype(xp.float32).reshape(out_f, in_f // group_size, group_size)
    wmin, wmax = g.min(2), g.max(2)
    scale = (wmax - wmin) / 15.0
    scale = xp.where(scale == 0, 1.0, scale).astype(xp.float32)
    zero = xp.clip(xp.rint(-wmin / scale), 0, 15).astype(xp.float32)
    q = xp.clip(xp.rint(g / scale[:, :, None] + zero[:, :, None]), 0, 15
                ).astype(xp.uint8).reshape(out_f, in_f)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(xp.uint8)  # (out, in//2)
    return {"qweight": xp.ascontiguousarray(packed),
            "scales": scale.astype(xp.float16),
            "zeros": zero.astype(xp.float16),
            "shape": (out_f, in_f), "group": group_size}


def dequantize_int4_rowmajor(payload, xp=np):
    out_f, in_f = payload["shape"]
    gs = payload["group"]
    p = payload["qweight"]
    lo = (p & 0xF).astype(xp.float32)
    hi = (p >> 4).astype(xp.float32)
    q = xp.empty((out_f, in_f), dtype=xp.float32)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    scale = xp.repeat(payload["scales"].astype(xp.float32), gs, axis=1)
    zero = xp.repeat(payload["zeros"].astype(xp.float32), gs, axis=1)
    return (q - zero) * scale


def _gemv_kernel(cp):
    """One thread per output row; x staged in shared memory. M=1 (decode)."""
    global _GEMV
    if _GEMV is None:
        _GEMV = cp.RawKernel(r"""
        #include <cuda_fp16.h>
        extern "C" __global__ void gemv_int4(
                const float* __restrict__ x,        // (in,)
                const unsigned char* __restrict__ qw, // (out, in/2)
                const __half* __restrict__ scales,  // (out, in/gs)
                const __half* __restrict__ zeros,   // (out, in/gs)
                float* __restrict__ y,              // (out,)
                int in_f, int out_f, int gs) {
            extern __shared__ float xs[];
            for (int i = threadIdx.x; i < in_f; i += blockDim.x) xs[i] = x[i];
            __syncthreads();
            int o = blockIdx.x * blockDim.x + threadIdx.x;
            if (o >= out_f) return;
            int ng = in_f / gs;
            const unsigned char* wr = qw + (long long)o * (in_f / 2);
            const __half* sc = scales + (long long)o * ng;
            const __half* ze = zeros + (long long)o * ng;
            float acc = 0.0f;
            for (int gi = 0; gi < ng; gi++) {
                float s = __half2float(sc[gi]);
                float z = __half2float(ze[gi]);
                int i0 = gi * gs;
                for (int t = 0; t < gs; t += 2) {
                    unsigned char b = wr[(i0 + t) / 2];
                    float q0 = (float)(b & 0xF);
                    float q1 = (float)(b >> 4);
                    acc += (q0 - z) * s * xs[i0 + t];
                    acc += (q1 - z) * s * xs[i0 + t + 1];
                }
            }
            y[o] = acc;
        }""", "gemv_int4")
    return _GEMV


def _quantize_int4_rowmajor_batched(W, group_size: int, xp):
    """Batched RTN group INT4 over a (E, out, in) tensor — same numerics as
    quantize_int4_rowmajor but for all E experts at once (a handful of GPU ops
    instead of an E-long Python loop). Returns (qweight (E,out,in/2) uint8,
    scales (E,out,in/gs) f16, zeros f16, (out,in))."""
    Ex, out_f, in_f = W.shape
    assert in_f % group_size == 0 and in_f % 2 == 0
    g = W.astype(xp.float32).reshape(Ex, out_f, in_f // group_size, group_size)
    wmin, wmax = g.min(3), g.max(3)
    scale = (wmax - wmin) / 15.0
    scale = xp.where(scale == 0, 1.0, scale).astype(xp.float32)
    zero = xp.clip(xp.rint(-wmin / scale), 0, 15).astype(xp.float32)
    q = xp.clip(xp.rint(g / scale[..., None] + zero[..., None]), 0, 15
                ).astype(xp.uint8).reshape(Ex, out_f, in_f)
    packed = (q[:, :, 0::2] | (q[:, :, 1::2] << 4)).astype(xp.uint8)
    return (xp.ascontiguousarray(packed), scale.astype(xp.float16),
            zero.astype(xp.float16), (out_f, in_f))


def build_stacked_experts(expert_ws, group_size: int = 128, xp=np) -> dict:
    """Stack E experts' {gate,up,down} into contiguous INT4 tensors indexable by
    expert id (for the fused selective kernel). expert_ws: list of dicts of fp16
    (out,in) weight matrices. Returns per-proj {qweight (E,out,in/2), scales
    (E,out,in/gs), zeros (E,out,in/gs)} plus dims.

    Quantization is **batched over all E experts on the GPU** — stack each proj
    into one (E,out,in) tensor and quantize it in one shot (vs the old per-expert
    Python loop of E*3 calls/layer, which dominated graph-MoE startup)."""
    out = {"group": group_size}
    for proj in ("gate", "up", "down"):
        stack = xp.asarray(np.stack([np.asarray(w[proj]) for w in expert_ws]))
        qw, sc, ze, shp = _quantize_int4_rowmajor_batched(stack, group_size, xp)
        out[f"{proj}.qweight"] = qw
        out[f"{proj}.scales"] = sc
        out[f"{proj}.zeros"] = ze
        out[f"{proj}.shape"] = shp
    return out


_GATE = None


def _gate_kernel(cp):
    """Capturable fp16 gate matvec: logits[e] = dot(x, gate_w[e]). One block per
    expert, threads over hidden + block reduction. Replaces the cuBLAS matmul
    (which can't run during CUDA-graph capture)."""
    global _GATE
    if _GATE is None:
        _GATE = cp.RawKernel(r"""
        #include <cuda_fp16.h>
        extern "C" __global__ void gate_matvec(
                const float* __restrict__ x, const __half* __restrict__ gw,
                float* __restrict__ logits, int hidden) {
            int e = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
            const __half* w = gw + (long long)e * hidden;
            float acc = 0.0f;
            for (int i = tid; i < hidden; i += nt) acc += __half2float(w[i]) * x[i];
            extern __shared__ float sh[];
            sh[tid] = acc; __syncthreads();
            for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) sh[tid] += sh[tid+s]; __syncthreads(); }
            if (tid == 0) logits[e] = sh[0];
        }""", "gate_matvec")
    return _GATE


def gate_matvec(x, gate_w, E, hidden, out=None, threads=128):
    """logits (E,) = x(hidden,) @ gate_w(E,hidden).T, capturable (no cuBLAS)."""
    import cupy as cp
    k = _gate_kernel(cp)
    x = cp.ascontiguousarray(x.astype(cp.float32).ravel())
    y = out if out is not None else cp.empty((E,), dtype=cp.float32)
    k((E,), (threads,), (x, gate_w, y, np.int32(hidden)), shared_mem=threads * 4)
    return y


_FUSEDW = None


def _fusedw_kernels(cp):
    """(E,)-routing-weight-driven selective MoE FFN kernels — grid over ALL E
    experts, each block early-outs if its weight is 0 (fully capturable, no D2H).

    Design (2026-07-20 rewrite, ~5x faster: 183->36us at Qwen3-30B shapes):
      * the input vector is staged **once per block in shared memory** — the old
        kernel re-read x from global for every one of ~6k output rows, and
        streaming the weights through L1 evicted x, so it ran at ~5% of BW; now
        each block loads x/inter once and reuses it for a TILE of rows;
      * **warp-per-row with a shuffle reduction** (no __syncthreads in the dot);
      * **vectorized 4-byte INT4 loads** (8 nibbles per load).
    ROWS_PER_BLOCK (rpb) is a launch arg so the grid can be sized for occupancy."""
    global _FUSEDW
    if _FUSEDW is None:
        src = r"""
        #include <cuda_fp16.h>
        // warp (32 lanes) dot-products one output row against `vec`; INT4 weights
        // read 4 bytes (8 nibbles) at a time; result valid on lane 0.
        __device__ __forceinline__ float wrd(
                const unsigned char* qw, const __half* sc, const __half* ze,
                int row, const float* vec, int in_f, int gs, int lane) {
            const unsigned char* wr = qw + (long long)row * (in_f / 2);
            const __half* s = sc + (long long)row * (in_f / gs);
            const __half* z = ze + (long long)row * (in_f / gs);
            float acc = 0.0f;
            for (int p = lane * 8; p < in_f; p += 32 * 8) {
                int gi = p / gs; float sv = __half2float(s[gi]), zv = __half2float(z[gi]);
                unsigned int packed = *reinterpret_cast<const unsigned int*>(wr + (p >> 1));
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    unsigned char b = (packed >> (j * 8)) & 0xFF;
                    acc += ((float)(b & 0xF) - zv) * sv * vec[p + 2 * j];
                    acc += ((float)(b >> 4) - zv) * sv * vec[p + 2 * j + 1];
                }
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
            return acc;
        }
        extern "C" __global__ void moew_gate_up(
                const float* __restrict__ x, const float* __restrict__ rw,
                const unsigned char* gqw, const __half* gsc, const __half* gze,
                const unsigned char* uqw, const __half* usc, const __half* uze,
                float* __restrict__ inter, int hidden, int ninter, int gs, int rpb) {
            int e = blockIdx.y; if (rw[e] == 0.0f) return;   // non-routed: skip
            int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, nw = blockDim.x >> 5;
            extern __shared__ float xs[];                    // hidden floats
            for (int i = tid; i < hidden; i += blockDim.x) xs[i] = x[i];
            __syncthreads();
            long long qb = (long long)e * ninter * (hidden / 2);
            long long sb = (long long)e * ninter * (hidden / gs);
            int tile = blockIdx.x * rpb;
            for (int r = tile + wid; r < tile + rpb && r < ninter; r += nw) {
                float g = wrd(gqw+qb, gsc+sb, gze+sb, r, xs, hidden, gs, lane);
                float u = wrd(uqw+qb, usc+sb, uze+sb, r, xs, hidden, gs, lane);
                if (lane == 0) inter[(long long)e*ninter + r] = (g/(1.0f+__expf(-g)))*u;
            }
        }
        extern "C" __global__ void moew_down(
                const float* __restrict__ inter, const float* __restrict__ rw,
                const unsigned char* dqw, const __half* dsc, const __half* dze,
                float* __restrict__ out, int hidden, int ninter, int gs, int rpb) {
            int e = blockIdx.y; float w = rw[e]; if (w == 0.0f) return;
            int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, nw = blockDim.x >> 5;
            extern __shared__ float is[];                    // ninter floats
            const float* iv = inter + (long long)e * ninter;
            for (int i = tid; i < ninter; i += blockDim.x) is[i] = iv[i];
            __syncthreads();
            long long qb = (long long)e * hidden * (ninter / 2);
            long long sb = (long long)e * hidden * (ninter / gs);
            int tile = blockIdx.x * rpb;
            for (int o = tile + wid; o < tile + rpb && o < hidden; o += nw) {
                float d = wrd(dqw+qb, dsc+sb, dze+sb, o, is, ninter, gs, lane);
                if (lane == 0) atomicAdd(&out[o], w * d);
            }
        }
        """
        _FUSEDW = (cp.RawKernel(src, "moew_gate_up"), cp.RawKernel(src, "moew_down"))
    return _FUSEDW


# vectorized INT4 loads require in_f % (32*8) == 0 (holds for hidden=2048,
# inter=768). rpb/threads tuned for occupancy at Qwen3-30B shapes (5090).
_MOE_RPB, _MOE_THREADS = 8, 256


def fused_moe_weighted(x, stacked, rw, E, hidden, inter, out=None, inter_buf=None,
                       threads=_MOE_THREADS, rpb=_MOE_RPB):
    """Capturable selective MoE FFN driven by an (E,) routing-weight vector.
    inter_buf must be (E, inter). Only rw[e]>0 experts contribute. x is staged in
    shared memory per block, so hidden and inter must fit (both are small)."""
    import cupy as cp
    gu, dn = _fusedw_kernels(cp)
    gs = stacked["group"]
    x = cp.ascontiguousarray(x.astype(cp.float32).ravel())
    rw = cp.ascontiguousarray(rw.astype(cp.float32).ravel())
    ibuf = inter_buf if inter_buf is not None else cp.empty((E, inter), dtype=cp.float32)
    y = out if out is not None else cp.zeros((hidden,), dtype=cp.float32)
    if out is not None:
        y.fill(0)
    gu(((inter + rpb - 1) // rpb, E), (threads,),
       (x, rw, stacked["gate.qweight"], stacked["gate.scales"], stacked["gate.zeros"],
        stacked["up.qweight"], stacked["up.scales"], stacked["up.zeros"],
        ibuf, np.int32(hidden), np.int32(inter), np.int32(gs), np.int32(rpb)),
       shared_mem=hidden * 4)
    dn(((hidden + rpb - 1) // rpb, E), (threads,),
       (ibuf, rw, stacked["down.qweight"], stacked["down.scales"], stacked["down.zeros"],
        y, np.int32(hidden), np.int32(inter), np.int32(gs), np.int32(rpb)),
       shared_mem=inter * 4)
    return y


_FUSED2 = None


def _fused2_kernels(cp):
    """Two-kernel fused MoE with proper parallelism: one block per (expert,
    output-row), threads cooperate over the input dim (block reduction). K1
    computes swiglu(gate,up) -> intermediate[K,inter]; K2 does down + weighted
    atomic-accumulate. Far more blocks than the one-block-per-expert version, so
    the GPU isn't idle."""
    global _FUSED2
    if _FUSED2 is None:
        src = r"""
        #include <cuda_fp16.h>
        __device__ __forceinline__ float row_dot(
                const unsigned char* qw, const __half* sc, const __half* ze,
                int row, const float* vec, int in_f, int gs, int tid, int nt) {
            const unsigned char* wr = qw + (long long)row * (in_f / 2);
            const __half* s = sc + (long long)row * (in_f / gs);
            const __half* z = ze + (long long)row * (in_f / gs);
            float acc = 0.0f;
            for (int i = tid * 2; i < in_f; i += nt * 2) {   // 2 nibbles/byte/thread
                int gi = i / gs;
                float sv = __half2float(s[gi]), zv = __half2float(z[gi]);
                unsigned char b = wr[i / 2];
                acc += ((float)(b & 0xF) - zv) * sv * vec[i];
                acc += ((float)(b >> 4) - zv) * sv * vec[i + 1];
            }
            return acc;
        }
        __device__ __forceinline__ float blk_reduce(float v, float* sh, int tid, int nt) {
            sh[tid] = v; __syncthreads();
            for (int s = nt >> 1; s > 0; s >>= 1) {
                if (tid < s) sh[tid] += sh[tid + s];
                __syncthreads();
            }
            return sh[0];
        }
        extern "C" __global__ void moe_gate_up(
                const float* __restrict__ x,
                const unsigned char* gqw, const __half* gsc, const __half* gze,
                const unsigned char* uqw, const __half* usc, const __half* uze,
                const int* __restrict__ eidx, float* __restrict__ inter,
                int hidden, int ninter, int gs) {
            int k = blockIdx.y, r = blockIdx.x;   // expert slot, inter row
            int e = eidx[k], tid = threadIdx.x, nt = blockDim.x;
            long long qb = (long long)e * ninter * (hidden / 2);
            long long sb = (long long)e * ninter * (hidden / gs);
            extern __shared__ float sh[];
            float gp = row_dot(gqw + qb, gsc + sb, gze + sb, r, x, hidden, gs, tid, nt);
            float g = blk_reduce(gp, sh, tid, nt);
            __syncthreads();
            float up = row_dot(uqw + qb, usc + sb, uze + sb, r, x, hidden, gs, tid, nt);
            float u = blk_reduce(up, sh, tid, nt);
            if (tid == 0) inter[(long long)k * ninter + r] = (g / (1.0f + __expf(-g))) * u;
        }
        extern "C" __global__ void moe_down(
                const float* __restrict__ inter,
                const unsigned char* dqw, const __half* dsc, const __half* dze,
                const int* __restrict__ eidx, const float* __restrict__ rw,
                float* __restrict__ out, int hidden, int ninter, int gs) {
            int k = blockIdx.y, o = blockIdx.x;   // expert slot, hidden row
            int e = eidx[k], tid = threadIdx.x, nt = blockDim.x;
            long long qb = (long long)e * hidden * (ninter / 2);
            long long sb = (long long)e * hidden * (ninter / gs);
            const float* iv = inter + (long long)k * ninter;
            extern __shared__ float sh[];
            float dp = row_dot(dqw + qb, dsc + sb, dze + sb, o, iv, ninter, gs, tid, nt);
            float d = blk_reduce(dp, sh, tid, nt);
            if (tid == 0) atomicAdd(&out[o], rw[k] * d);
        }
        """
        _FUSED2 = (cp.RawKernel(src, "moe_gate_up"), cp.RawKernel(src, "moe_down"))
    return _FUSED2


def fused_moe_ffn2(x, stacked, eidx, rw, hidden, inter, out=None, inter_buf=None,
                   threads=128):
    """Efficient two-kernel selective MoE FFN (see _fused2_kernels)."""
    import cupy as cp

    gu, dn = _fused2_kernels(cp)
    gs = stacked["group"]
    K = int(eidx.shape[0])
    x = cp.ascontiguousarray(x.astype(cp.float32).ravel())
    eidx = eidx.astype(cp.int32)
    rw = rw.astype(cp.float32)
    ibuf = inter_buf if inter_buf is not None else cp.empty((K, inter), dtype=cp.float32)
    y = out if out is not None else cp.zeros((hidden,), dtype=cp.float32)
    if out is not None:
        y.fill(0)
    gu((inter, K), (threads,),
       (x, stacked["gate.qweight"], stacked["gate.scales"], stacked["gate.zeros"],
        stacked["up.qweight"], stacked["up.scales"], stacked["up.zeros"],
        eidx, ibuf, np.int32(hidden), np.int32(inter), np.int32(gs)),
       shared_mem=threads * 4)
    dn((hidden, K), (threads,),
       (ibuf, stacked["down.qweight"], stacked["down.scales"], stacked["down.zeros"],
        eidx, rw, y, np.int32(hidden), np.int32(inter), np.int32(gs)),
       shared_mem=threads * 4)
    return y


_FUSED = None


def _fused_kernel(cp):
    """One block per routed expert: gate·up·swiglu into shared, then down,
    accumulate routing_weight * result into the output. Device expert indices,
    so it's selective (only routed experts read) and capturable."""
    global _FUSED
    if _FUSED is None:
        _FUSED = cp.RawKernel(r"""
        #include <cuda_fp16.h>
        __device__ __forceinline__ float row_gemv(
                const unsigned char* qw, const __half* sc, const __half* ze,
                int row, const float* vec, int in_f, int gs) {
            int ng = in_f / gs;
            const unsigned char* wr = qw + (long long)row * (in_f / 2);
            const __half* s = sc + (long long)row * ng;
            const __half* z = ze + (long long)row * ng;
            float acc = 0.0f;
            for (int gi = 0; gi < ng; gi++) {
                float sv = __half2float(s[gi]), zv = __half2float(z[gi]);
                int i0 = gi * gs;
                for (int t = 0; t < gs; t += 2) {
                    unsigned char b = wr[(i0 + t) / 2];
                    acc += ((float)(b & 0xF) - zv) * sv * vec[i0 + t];
                    acc += ((float)(b >> 4) - zv) * sv * vec[i0 + t + 1];
                }
            }
            return acc;
        }
        extern "C" __global__ void fused_moe(
                const float* __restrict__ x,          // (hidden,)
                const unsigned char* __restrict__ gqw, const __half* gsc, const __half* gze,
                const unsigned char* __restrict__ uqw, const __half* usc, const __half* uze,
                const unsigned char* __restrict__ dqw, const __half* dsc, const __half* dze,
                const int* __restrict__ eidx,         // (K,) routed expert ids
                const float* __restrict__ rw,         // (K,) routing weights
                float* __restrict__ out,              // (hidden,) accumulated
                int hidden, int inter, int gs) {
            int k = blockIdx.x;
            int e = eidx[k];
            float w = rw[k];
            int tid = threadIdx.x, nt = blockDim.x;
            extern __shared__ float sh[];
            float* xs = sh;              // hidden
            float* is = sh + hidden;     // inter
            for (int i = tid; i < hidden; i += nt) xs[i] = x[i];
            __syncthreads();
            // per-expert weight bases
            long long gq = (long long)e * inter * (hidden / 2);
            long long gsz = (long long)e * inter * (hidden / gs);
            long long dq = (long long)e * hidden * (inter / 2);
            long long dsz = (long long)e * hidden * (inter / gs);
            for (int r = tid; r < inter; r += nt) {
                float gr = row_gemv(gqw + gq, gsc + gsz, gze + gsz, r, xs, hidden, gs);
                float ur = row_gemv(uqw + gq, usc + gsz, uze + gsz, r, xs, hidden, gs);
                is[r] = (gr / (1.0f + __expf(-gr))) * ur;   // silu(gate)*up
            }
            __syncthreads();
            for (int o = tid; o < hidden; o += nt) {
                float dv = row_gemv(dqw + dq, dsc + dsz, dze + dsz, o, is, inter, gs);
                atomicAdd(&out[o], w * dv);
            }
        }""", "fused_moe")
    return _FUSED


def fused_moe_ffn(x, stacked, eidx, rw, hidden, inter, out=None):
    """Selective MoE FFN: sum_k rw[k] * down_e(swiglu(gate_e(x), up_e(x))) for
    e = eidx[k], in one launch. x: (hidden,) fp32; eidx: (K,) int32; rw: (K,)
    fp32; stacked: build_stacked_experts output on GPU. Returns (hidden,) fp32."""
    import cupy as cp

    k = _fused_kernel(cp)
    gs = stacked["group"]
    K = int(eidx.shape[0])
    x = cp.ascontiguousarray(x.astype(cp.float32).ravel())
    y = out if out is not None else cp.zeros((hidden,), dtype=cp.float32)
    if out is not None:
        y.fill(0)
    threads = 256
    k((K,), (threads,),
      (x, stacked["gate.qweight"], stacked["gate.scales"], stacked["gate.zeros"],
       stacked["up.qweight"], stacked["up.scales"], stacked["up.zeros"],
       stacked["down.qweight"], stacked["down.scales"], stacked["down.zeros"],
       eidx.astype(cp.int32), rw.astype(cp.float32), y,
       np.int32(hidden), np.int32(inter), np.int32(gs)),
      shared_mem=(hidden + inter) * 4)
    return y


def gemv_int4(x, payload, out=None):
    """y = x @ dequant(W).T for a single decode row. x: (in,) fp32 cupy.
    payload: quantize_int4_rowmajor output on GPU. Returns (out,) fp32."""
    import cupy as cp

    k = _gemv_kernel(cp)
    out_f, in_f = payload["shape"]
    gs = payload["group"]
    x = cp.ascontiguousarray(x.astype(cp.float32).ravel())
    y = out if out is not None else cp.empty((out_f,), dtype=cp.float32)
    threads = 128
    blocks = (out_f + threads - 1) // threads
    k((blocks,), (threads,),
      (x, payload["qweight"], payload["scales"], payload["zeros"], y,
       np.int32(in_f), np.int32(out_f), np.int32(gs)),
      shared_mem=in_f * 4)
    return y
