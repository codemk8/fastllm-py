"""ctypes wrapper for fastllm's FP8-E4M3 block-128 GEMV (small-batch decode path).

Backed by ``native/libfastllm_kernels.so`` (built by ``native/build.sh``), which
compiles ``native/fastllm_fp8_block128.cu`` -- a raw-pointer extraction of the
kernels in ``fastllm/src/devices/cuda/linear/fastllm-linear-fp8.cu`` (no cuBLAS,
no fastllm::Data; see docs/fastllm-internals.md §7.2).

Weight format wrapped: ``DataType::FP8_E4M3_BLOCK_128`` -- the INTERLEAVED,
inline-scale layout (docs §3.1-3.3), NOT the separate-scale-tensor generic FP8
format (§3.4). Each weight row of ``m`` input features is stored as::

    [fp8_0 .. fp8_127][float32 scale_0][fp8_128 .. fp8_255][float32 scale_1] ...

i.e. one 128-value FP8-E4M3 block followed inline by its float32 block scale,
repeated ``m/128`` times. Bytes per row = ``m + (m/128)*4`` = ``perRow``.
The scale for a block is ``max(|w|)/448`` over that block; values are genuine
hardware E4M3 (not int8). Use ``quantize`` (which calls the exact fastllm CUDA
quantize kernel) to produce this layout so it is bit-identical to the engine.

GEMV semantics: ``C[n, k] = A[n, m] @ dequant(W).T`` where logical weight ``W`` is
``[k, m]`` (k output rows, each ``perRow`` bytes). Optimized for small ``n``
(decode). Activations/outputs are fp16, row-major.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    here = os.path.dirname(os.path.abspath(__file__))
    so = os.path.abspath(os.path.join(here, "..", "..", "native", "libfastllm_kernels.so"))
    if not os.path.exists(so):
        raise FileNotFoundError(
            f"libfastllm_kernels.so not found at {so}; run native/build.sh")
    lib = ctypes.CDLL(so)

    lib.fastllm_fp8_block128_quantize.restype = ctypes.c_int
    lib.fastllm_fp8_block128_quantize.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    lib.fastllm_fp8_block128_packed_row_bytes.restype = ctypes.c_int
    lib.fastllm_fp8_block128_packed_row_bytes.argtypes = [ctypes.c_int]

    lib.fastllm_fp8_block128_gemv_fp16.restype = ctypes.c_int
    lib.fastllm_fp8_block128_gemv_fp16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _lib = lib
    return lib


def available() -> bool:
    try:
        _load()
        return True
    except Exception:
        return False


def packed_row_bytes(m: int) -> int:
    """Bytes per weight row in FP8_E4M3_BLOCK_128 layout = m + (m/128)*4."""
    return _load().fastllm_fp8_block128_packed_row_bytes(int(m))


def quantize(weight_fp16, is_bf16: bool = False):
    """Quantize a [k, m] fp16/bf16 weight (cupy, device-resident) to block128.

    weight_fp16 : cupy [k(out), m(in)] fp16 (or bf16 if is_bf16). ``m % 128 == 0``.
    Returns a cupy uint8 array of shape ``[k, perRow]`` (perRow = packed_row_bytes(m))
    holding the interleaved FP8 + inline float32 scales -- bit-identical to fastllm's
    FastllmQuantizeLinearWeightFP8E4M3Block128Kernel.
    """
    import cupy as cp
    lib = _load()
    k, m = weight_fp16.shape
    if m % 128 != 0:
        raise ValueError(f"in-features m ({m}) must be a multiple of 128")
    weight_fp16 = cp.ascontiguousarray(weight_fp16)
    per_row = packed_row_bytes(m)
    out = cp.empty((k, per_row), dtype=cp.uint8)
    rc = lib.fastllm_fp8_block128_quantize(
        ctypes.c_void_p(int(weight_fp16.data.ptr)),
        ctypes.c_void_p(int(out.data.ptr)),
        ctypes.c_int(k), ctypes.c_int(m), ctypes.c_int(1 if is_bf16 else 0))
    if rc != 0:
        raise RuntimeError(f"fastllm_fp8_block128_quantize failed (rc={rc})")
    return out


def fp8_gemv(x_fp16, weight_fp8_block128, scales=None, out=None, k=None):
    """FP8 block-128 GEMV/GEMM.

    x_fp16              : cupy fp16 [n, m] row-major activations.
    weight_fp8_block128 : cupy uint8 weight in block128 layout. Either shape
                          ``[k, perRow]`` (preferred, k inferred) or a flat array
                          (then ``k`` must be passed). Inline block scales live
                          INSIDE this buffer -- there is no separate scale tensor
                          for this format, so ``scales`` is ignored/None.
    out                 : optional cupy fp16 [n, k] output buffer.
    k                   : output features; required only if weight is 1-D.

    Returns cupy fp16 [n, k] = x_fp16 @ dequant(weight).T.
    """
    import cupy as cp
    lib = _load()
    if scales is not None:
        raise ValueError("FP8_E4M3_BLOCK_128 stores scales inline; pass scales=None")

    x_fp16 = cp.ascontiguousarray(x_fp16.astype(cp.float16))
    n, m = x_fp16.shape
    per_row = packed_row_bytes(m)

    w = cp.ascontiguousarray(weight_fp8_block128)
    if w.ndim == 2:
        k = w.shape[0]
        if w.shape[1] != per_row:
            raise ValueError(f"weight row bytes {w.shape[1]} != expected perRow {per_row}")
    else:
        if k is None:
            raise ValueError("k must be given when weight is 1-D")
        if w.size != k * per_row:
            raise ValueError("flat weight size does not match k*perRow")

    if out is None:
        out = cp.empty((n, k), dtype=cp.float16)
    rc = lib.fastllm_fp8_block128_gemv_fp16(
        ctypes.c_void_p(int(x_fp16.data.ptr)),
        ctypes.c_void_p(int(w.data.ptr)),
        ctypes.c_void_p(int(out.data.ptr)),
        ctypes.c_void_p(0),  # bias
        ctypes.c_int(n), ctypes.c_int(m), ctypes.c_int(k), ctypes.c_int(per_row))
    if rc != 0:
        raise RuntimeError(f"fastllm_fp8_block128_gemv_fp16 failed (rc={rc})")
    return out
