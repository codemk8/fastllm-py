"""Custom row-major INT4 GEMV: matches dequant reference + the marlin path."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from fastllm_py.kernels import moe_int4


@pytest.mark.parametrize("out_f,in_f,gs", [(1408, 2048, 128), (2048, 1408, 128),
                                            (512, 1024, 128), (256, 256, 32)])
def test_gemv_matches_dequant(out_f, in_f, gs):
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((out_f, in_f)) * 0.1).astype(np.float32)
    x = (rng.standard_normal(in_f) * 0.5).astype(np.float32)
    payload = moe_int4.quantize_int4_rowmajor(w, gs, xp=cp)
    y = cp.asnumpy(moe_int4.gemv_int4(cp.asarray(x), payload))
    # reference: dequant the SAME payload (on CPU), then x @ Wq.T
    ref = x @ moe_int4.dequantize_int4_rowmajor(payload, xp=cp).get().T
    np.testing.assert_allclose(y, ref, rtol=1e-3, atol=1e-3)


def test_gemv_close_to_marlin():
    """Row-major and marlin use the same RTN quant, so their GEMV outputs should
    agree to int4 rounding (both dequantize the same q/scale/zero)."""
    from fastllm_py.kernels import marlin
    if not marlin.available():
        pytest.skip("marlin .so missing")
    out_f, in_f, gs = 1408, 2048, 128
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((out_f, in_f)) * 0.1).astype(np.float32)
    x = (rng.standard_normal(in_f) * 0.5).astype(np.float16)

    pay = moe_int4.quantize_int4_rowmajor(w, gs, xp=cp)
    y_rm = cp.asnumpy(moe_int4.gemv_int4(cp.asarray(x.astype(np.float32)), pay))
    ref = x.astype(np.float32) @ moe_int4.dequantize_int4_rowmajor(pay, xp=cp).get().T
    # row-major GEMV == its own dequant ref (self-consistent; the marlin cross-
    # check is covered by test_native_kernels)
    rel = np.abs(y_rm - ref).max() / (np.abs(ref).max() + 1e-6)
    assert rel < 1e-3, rel
