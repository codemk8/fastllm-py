"""Native CUDA kernel tests: Marlin INT4 GEMM + FP8-E4M3 block-128 GEMV.

Exercises native/libfastllm_kernels.so end-to-end:
  * quantize a random weight matrix with the *same* algorithm fastllm uses
    (Marlin: numpy int4-group port; FP8: the exposed fastllm CUDA quantize kernel),
  * run the native GEMM/GEMV against cupy fp16 activations,
  * compare to x @ dequant(W).T with a relative-Frobenius tolerance.

Skips cleanly when the .so is missing or no CUDA/cupy is available.
"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from fastllm_py.kernels import fp8, marlin  # noqa: E402
from fastllm_py.quantizer import _fp8_e4m3_decode, _fp8_e4m3_encode  # noqa: E402


def _have_gpu():
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _have_gpu(), reason="no CUDA device")


def _rel_fro(a, b):
    a = a.astype(cp.float32)
    b = b.astype(cp.float32)
    return float(cp.linalg.norm(a - b) / (cp.linalg.norm(b) + 1e-12))


# ---------------------------------------------------------------------------
# FP8-E4M3 block-128 GEMV
# ---------------------------------------------------------------------------
def _fp8_reference(W, x):
    """W [k, m] fp32, x [n, m] fp32 -> cupy fp16 x @ dequant(W).T.

    Reproduces fastllm's per-128-column-block FP8-E4M3 weight quantization
    (scale = max(|w|)/448 per row-block) using the validated E4M3 codec.
    """
    k, m = W.shape
    Wr = W.reshape(k, m // 128, 128)
    amax = np.abs(Wr).max(axis=2, keepdims=True)
    scale = np.where(amax == 0, 1.0, amax / 448.0)
    q = _fp8_e4m3_encode(Wr / scale, np)
    deq = (_fp8_e4m3_decode(q, np) * scale).reshape(k, m)
    return cp.asarray(x @ deq.T, dtype=cp.float16)


@pytest.mark.skipif(not fp8.available(), reason="libfastllm_kernels.so (FP8) missing")
@pytest.mark.parametrize("n", [1, 16])           # m=1 GEMV path, m=16 GEMM path
@pytest.mark.parametrize("m,k", [
    (1024, 512),      # generic dense
    (2048, 1408),     # MoE-expert-like (k output = 1408)
    (1408, 2048),     # MoE-expert-like (m input = 1408)
])
def test_fp8_block128_gemv(n, m, k):
    rng = np.random.default_rng(1234 + n)
    W = (rng.standard_normal((k, m)).astype(np.float32)) * 0.1
    x = (rng.standard_normal((n, m)).astype(np.float32)) * 0.5

    wq = fp8.quantize(cp.asarray(W, dtype=cp.float16))       # exact fastllm kernel
    assert wq.shape == (k, fp8.packed_row_bytes(m))
    out = fp8.fp8_gemv(cp.asarray(x, dtype=cp.float16), wq)
    assert out.shape == (n, k)

    ref = _fp8_reference(W, x)
    err = _rel_fro(out, ref)
    assert err < 0.05, f"FP8 rel-Frobenius error too high: {err}"


# ---------------------------------------------------------------------------
# Marlin INT4 (uint4, AWQ-zero-point) GEMM
# ---------------------------------------------------------------------------
def _quant_int4_group(W, gs):
    """Asymmetric int4-group quant of W [size_n, size_k] along size_k.

    Returns (q uint8 [n,k], scale [n,g], zero_int [n,g], W_hat [n,k]) where
    W_hat = scale * (q - zero_int) -- exactly what Marlin dequantizes.
    """
    size_n, size_k = W.shape
    g = W.reshape(size_n, size_k // gs, gs)
    wmin = g.min(axis=2)
    wmax = g.max(axis=2)
    scale = (wmax - wmin) / 15.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    zero = np.clip(np.rint(-wmin / scale), 0, 15).astype(np.int64)
    q = np.clip(np.rint((g - wmin[:, :, None]) / scale[:, :, None]), 0, 15).astype(np.uint8)
    q = q.reshape(size_n, size_k)
    W_hat = (scale[:, :, None]
             * (q.reshape(size_n, size_k // gs, gs).astype(np.float32) - zero[:, :, None])
             ).reshape(size_n, size_k)
    return q, scale, zero, W_hat


@pytest.mark.skipif(not marlin.available(), reason="libfastllm_kernels.so (Marlin) missing")
@pytest.mark.parametrize("size_m", [1, 16])       # 1 = GEMV path, 16 = GEMM path
@pytest.mark.parametrize("size_k,size_n", [
    (1024, 512),      # generic dense
    (2048, 1408),     # MoE-expert-like output dim 1408
    (1408, 2048),     # MoE-expert-like input dim 1408
])
@pytest.mark.parametrize("gs", [128, 32])
def test_marlin_gemm_int4(size_m, size_k, size_n, gs):
    rng = np.random.default_rng(99 + size_m + size_k + size_n + gs)
    W = (rng.standard_normal((size_n, size_k)).astype(np.float32)) * 0.1
    a = (rng.standard_normal((size_m, size_k)).astype(np.float32)) * 0.5

    q, scale, zero, W_hat = _quant_int4_group(W, gs)

    std = marlin.pack_gptq_qweight(cp.asarray(q), xp=cp)
    packed = marlin.marlin_repack(std, size_k, size_n)
    scales, zeros = marlin.build_marlin_scales_zeros(
        cp.asarray(scale), cp.asarray(zero), gs, xp=cp)
    ws = marlin.make_workspace(size_n)

    out = marlin.marlin_gemm_int4(cp.asarray(a, dtype=cp.float16),
                                  packed, scales, zeros, ws)
    assert out.shape == (size_m, size_n)

    ref = cp.asarray(a @ W_hat.T, dtype=cp.float16)
    err = _rel_fro(out, ref)
    # Reference uses the exact same q/scale/zero, so this isolates repack +
    # permutation + kernel correctness -> should be near-exact.
    assert err < 0.02, f"Marlin rel-Frobenius error too high: {err}"


@pytest.mark.skipif(not marlin.available() or not marlin.has_stream_gemm(),
                    reason="stream-accepting Marlin entry missing")
@pytest.mark.parametrize("size_m", [1, 16])
def test_marlin_gemm_stream_matches_default(size_m):
    """The stream-accepting entry must be bit-identical to the default one."""
    size_k, size_n, gs = 2048, 1408, 128
    rng = np.random.default_rng(7)
    W = (rng.standard_normal((size_n, size_k)).astype(np.float32)) * 0.1
    a = (rng.standard_normal((size_m, size_k)).astype(np.float32)) * 0.5
    q, scale, zero, _ = _quant_int4_group(W, gs)
    packed = marlin.marlin_repack(marlin.pack_gptq_qweight(cp.asarray(q), xp=cp),
                                  size_k, size_n)
    scales, zeros = marlin.build_marlin_scales_zeros(
        cp.asarray(scale), cp.asarray(zero), gs, xp=cp)
    a16 = cp.asarray(a, dtype=cp.float16)

    default = marlin.gemm_fast(a16, packed, scales, zeros, size_n, size_k)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        streamed = marlin.gemm_fast(a16, packed, scales, zeros, size_n, size_k,
                                    stream=stream)
    stream.synchronize()
    cp.testing.assert_array_equal(default, streamed)
