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
