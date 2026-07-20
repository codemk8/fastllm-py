"""ctypes wrapper for fastllm's Marlin INT4 (uint4, AWQ-zero-point) GEMM.

Backed by ``native/libfastllm_kernels.so`` (built by ``native/build.sh``), which
compiles ``fastllm/src/devices/cuda/linear/fastllm-marlin.cu`` unmodified.

Naming follows Marlin's own ``(size_m, size_n, size_k)`` convention:

* ``size_m`` = activation rows (batch)
* ``size_k`` = input features   (fastllm calls this ``m``; the GEMM reduction dim,
  and the axis quantization groups run along)
* ``size_n`` = output features  (fastllm calls this ``k``)

The GEMM computes ``C[size_m, size_n] = A[size_m, size_k] @ dequant(W).T`` where the
logical weight ``W`` is ``[size_n, size_k]`` int4, dequantized per group as
``scale * (q - zero)`` (see docs/fastllm-internals.md §2).

Weight preparation (done once, offline / at load time) has three device-independent
NumPy steps reproduced here from fastllm's host code:

1. ``pack_gptq_qweight`` : int4 weight ``[size_n, size_k]`` -> "standard GPTQ" uint32
   packing (8 values-along-K per word), shape ``[size_k//8, size_n]``
   (origin: FastllmCudaInt4GroupToMarlinQWeightKernel, fastllm-linear-int4group.cu:850).
2. ``marlin_repack`` : device repack of that into Marlin's tiled register layout
   (origin: FastllmCudaGptqMarlinRepack, fastllm-marlin.cu:2637).
3. ``build_marlin_scales_zeros`` : scale/zero permutation into Marlin's tile layout
   (origin: FastllmBuildMarlinPermutedScalesAndZeros, fastllm-linear-int4group.cu:890).

Preconditions (enforced): ``size_n % 64 == 0``, ``size_k % 64 == 0``,
``group_size in {32, 128}``, ``size_k % group_size == 0``, device sm_75+.
The ``workspace`` int32 buffer must be zeroed before first use (see ``make_workspace``).
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

# Marlin's fixed tensor-core tile permutation tables
# (origin fastllm-linear-int4group.cu:898-908).
_SCALE_PERM = np.array([
    0, 8, 16, 24, 32, 40, 48, 56,
    1, 9, 17, 25, 33, 41, 49, 57,
    2, 10, 18, 26, 34, 42, 50, 58,
    3, 11, 19, 27, 35, 43, 51, 59,
    4, 12, 20, 28, 36, 44, 52, 60,
    5, 13, 21, 29, 37, 45, 53, 61,
    6, 14, 22, 30, 38, 46, 54, 62,
    7, 15, 23, 31, 39, 47, 55, 63,
], dtype=np.int64)
_ZP_INTERLEAVE = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int64)

_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    here = os.path.dirname(os.path.abspath(__file__))
    so = os.path.join(here, "..", "..", "native", "libfastllm_kernels.so")
    so = os.path.abspath(so)
    if not os.path.exists(so):
        raise FileNotFoundError(
            f"libfastllm_kernels.so not found at {so}; run native/build.sh")
    lib = ctypes.CDLL(so)

    lib.FastllmCudaGptqMarlinRepack.restype = ctypes.c_bool
    lib.FastllmCudaGptqMarlinRepack.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

    lib.FastllmCudaMarlinHalfInt4Gemm.restype = ctypes.c_bool
    lib.FastllmCudaMarlinHalfInt4Gemm.argtypes = [
        ctypes.c_void_p,  # a (fp16)
        ctypes.c_void_p,  # b_q_weight (uint32, marlin repacked)
        ctypes.c_void_p,  # b_scales (fp16, permuted)
        ctypes.c_void_p,  # b_zeros (uint32, permuted+packed)
        ctypes.c_void_p,  # c (fp16)
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # size_m, size_n, size_k
        ctypes.c_int,     # group_size
        ctypes.c_void_p,  # workspace (int32)
    ]
    _lib = lib
    return lib


def available() -> bool:
    try:
        _load()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# NumPy weight-prep helpers (host side, run once per weight)
# --------------------------------------------------------------------------
def pack_gptq_qweight(q_int4, xp=np):
    """int4 weight [size_n, size_k] (uint8 values 0..15) -> uint32 [size_k//8, size_n].

    Packs 8 consecutive along-K values into one word, value i at bit 4*i.
    Mirrors FastllmCudaInt4GroupToMarlinQWeightKernel (fastllm-linear-int4group.cu:850).
    """
    size_n, size_k = q_int4.shape
    assert size_k % 8 == 0, "size_k must be a multiple of 8"
    q = q_int4.astype(xp.uint32)
    # [size_n, size_k//8, 8]
    q = q.reshape(size_n, size_k // 8, 8)
    shifts = (xp.arange(8, dtype=xp.uint32) * 4)
    words = (q << shifts[None, None, :]).sum(axis=2).astype(xp.uint32)  # [size_n, size_k//8]
    # transpose to [size_k//8, size_n], row-major
    return xp.ascontiguousarray(words.T)


def build_marlin_scales_zeros(scale, zero_int, group_size, xp=np):
    """Permute scales/zeros into Marlin's tile layout.

    scale     : float [size_n, num_groups]  (per output-row, per K-group)
    zero_int  : int   [size_n, num_groups]  (0..15 zero points)
    returns (scales_fp16 [num_groups, size_n], zeros_u32 [num_groups, size_n//8]).
    Mirrors FastllmBuildMarlinPermutedScalesAndZeros (fastllm-linear-int4group.cu:890).
    """
    size_n, num_groups = scale.shape
    assert size_n % 64 == 0, "size_n must be a multiple of 64"
    # transpose to [num_groups, size_n]  (scaleGN[g, out])
    sgn = xp.ascontiguousarray(scale.T).astype(xp.float32)
    zgn = xp.ascontiguousarray(zero_int.T).astype(xp.int64)

    # scalePerm on every contiguous 64-chunk of the flat [num_groups*size_n] array.
    # size_n % 64 == 0 => chunks align inside each group row.
    perm = _SCALE_PERM if xp is np else xp.asarray(_SCALE_PERM)
    zpi = _ZP_INTERLEAVE if xp is np else xp.asarray(_ZP_INTERLEAVE)

    s3 = sgn.reshape(num_groups, size_n // 64, 64)
    scales = s3[:, :, perm].reshape(num_groups, size_n).astype(xp.float16)

    z3 = zgn.reshape(num_groups, size_n // 64, 64)
    z_perm = z3[:, :, perm].reshape(num_groups, size_n)  # scalePerm applied
    # then zpInterleave on 8-chunks, pack 8 nibbles per uint32
    z8 = z_perm.reshape(num_groups, size_n // 8, 8)
    z8 = z8[:, :, zpi].astype(xp.uint32)
    shifts = (xp.arange(8, dtype=xp.uint32) * 4)
    zeros = (z8 << shifts[None, None, :]).sum(axis=2).astype(xp.uint32)  # [num_groups, size_n//8]
    return xp.ascontiguousarray(scales), xp.ascontiguousarray(zeros)


def make_workspace(size_n, xp=None):
    """Zeroed int32 workspace, size max(1, (size_n//64)*16) (cupy device array)."""
    if xp is None:
        import cupy as xp
    n = max(1, (size_n // 64) * 16)
    return xp.zeros(n, dtype=xp.int32)


# --------------------------------------------------------------------------
# Kernel calls
# --------------------------------------------------------------------------
def marlin_repack(std_qweight, size_k, size_n):
    """Device repack. std_qweight: uint32 cupy array [size_k//8, size_n].

    Returns marlin-repacked uint32 cupy array (same element count).
    """
    import cupy as cp
    lib = _load()
    std_qweight = cp.ascontiguousarray(std_qweight.astype(cp.uint32))
    out = cp.empty(std_qweight.size, dtype=cp.uint32)
    ok = lib.FastllmCudaGptqMarlinRepack(
        ctypes.c_void_p(int(std_qweight.data.ptr)),
        ctypes.c_void_p(int(out.data.ptr)),
        ctypes.c_int(size_k), ctypes.c_int(size_n))
    cp.cuda.runtime.deviceSynchronize()
    if not ok:
        raise RuntimeError("FastllmCudaGptqMarlinRepack failed (check size_k/size_n tiling)")
    return out


_workspaces: dict = {}  # (device_id, size_n) -> zeroed int32 workspace


def get_workspace(size_n: int):
    """Shared per-device workspace (Marlin self-resets it between calls)."""
    import cupy as cp

    key = (cp.cuda.Device().id, size_n)
    if key not in _workspaces:
        _workspaces[key] = make_workspace(size_n, cp)
    return _workspaces[key]


def marlin_gemm_int4(a_fp16, packed_weights, scales, zeros, workspace=None,
                     size_n=None, sync=True):
    """Marlin uint4 GEMM.

    a_fp16          : cupy fp16 [size_m, size_k]  (row-major activations)
    packed_weights  : cupy uint32, marlin-repacked (from marlin_repack)
    scales          : cupy fp16 [num_groups, size_n]  (permuted, from build_marlin_scales_zeros)
    zeros           : cupy uint32 [num_groups, size_n//8]  (permuted+packed)
    workspace       : cupy int32, zeroed (from make_workspace(size_n))
    size_n          : output features; inferred from scales.shape[1] if None.

    Returns cupy fp16 [size_m, size_n].
    """
    import cupy as cp
    lib = _load()
    a_fp16 = cp.ascontiguousarray(a_fp16.astype(cp.float16))
    size_m, size_k = a_fp16.shape
    num_groups = scales.shape[0]
    if size_n is None:
        size_n = scales.shape[1]
    if workspace is None:
        workspace = get_workspace(size_n)
    group_size = size_k // num_groups

    if size_n % 64 != 0 or size_k % 64 != 0:
        raise ValueError(f"size_n({size_n}) and size_k({size_k}) must be multiples of 64")
    if group_size not in (32, 128):
        raise ValueError(f"group_size must be 32 or 128, got {group_size}")
    if size_k % group_size != 0:
        raise ValueError("size_k must be a multiple of group_size")

    scales = cp.ascontiguousarray(scales.astype(cp.float16))
    zeros = cp.ascontiguousarray(zeros.astype(cp.uint32))
    c = cp.empty((size_m, size_n), dtype=cp.float16)

    ok = lib.FastllmCudaMarlinHalfInt4Gemm(
        ctypes.c_void_p(int(a_fp16.data.ptr)),
        ctypes.c_void_p(int(packed_weights.data.ptr)),
        ctypes.c_void_p(int(scales.data.ptr)),
        ctypes.c_void_p(int(zeros.data.ptr)),
        ctypes.c_void_p(int(c.data.ptr)),
        ctypes.c_int(size_m), ctypes.c_int(size_n), ctypes.c_int(size_k),
        ctypes.c_int(group_size),
        ctypes.c_void_p(int(workspace.data.ptr)))
    if sync:  # launches go to the legacy null stream: stream-ordered vs
        cp.cuda.runtime.deviceSynchronize()  # blocking streams either way
    if not ok:
        raise RuntimeError("FastllmCudaMarlinHalfInt4Gemm returned false "
                           "(unsupported device or shape)")
    return c
