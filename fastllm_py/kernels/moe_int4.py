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


def build_stacked_experts(expert_ws, group_size: int = 128, xp=np) -> dict:
    """Stack E experts' {gate,up,down} into contiguous INT4 tensors indexable by
    expert id (for the fused selective kernel). expert_ws: list of dicts of fp16
    (out,in) weight matrices. Returns per-proj {qweight (E,out,in/2), scales
    (E,out,in/gs), zeros (E,out,in/gs)} plus dims."""
    out = {"group": group_size}
    for proj in ("gate", "up", "down"):
        qs, ss, zs = [], [], []
        for w in expert_ws:
            p = quantize_int4_rowmajor(w[proj], group_size, xp=xp)
            qs.append(p["qweight"]); ss.append(p["scales"]); zs.append(p["zeros"])
            out[f"{proj}.shape"] = p["shape"]
        out[f"{proj}.qweight"] = xp.ascontiguousarray(xp.stack(qs))
        out[f"{proj}.scales"] = xp.ascontiguousarray(xp.stack(ss))
        out[f"{proj}.zeros"] = xp.ascontiguousarray(xp.stack(zs))
    return out


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
