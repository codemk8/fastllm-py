import numpy as np
import pytest

from fastllm_py.quantizer import (
    dequantize_fp8_block, dequantize_int4_group,
    quantize_fp8_block, quantize_int4_group, _fp8_e4m3_encode, _fp8_e4m3_decode,
)


def test_fp8_e4m3_roundtrip_exact_values():
    # values exactly representable in e4m3
    vals = np.array([0.0, 0.5, 1.0, 1.5, 2.0, -3.5, 448.0, -448.0,
                     2.0**-6, 2.0**-9, 0.875], dtype=np.float32)
    enc = _fp8_e4m3_encode(vals, np)
    dec = _fp8_e4m3_decode(enc, np)
    np.testing.assert_allclose(dec, vals, rtol=0, atol=0)


def test_fp8_e4m3_vs_torch():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch without fp8")
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(4096).astype(np.float32) * 100).clip(-448, 448)
    ours = _fp8_e4m3_decode(_fp8_e4m3_encode(x, np), np)
    ref = torch.tensor(x).to(torch.float8_e4m3fn).float().numpy()
    np.testing.assert_allclose(ours, ref, rtol=0, atol=0)


def test_fp8_block_quant_error():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((256, 384)).astype(np.float32)
    qt = quantize_fp8_block(w, 128)
    wd = dequantize_fp8_block(qt)
    assert wd.shape == w.shape
    rel = np.abs(wd - w).mean() / np.abs(w).mean()
    assert rel < 0.05, rel


def test_fp8_block_nondivisible():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((100, 200)).astype(np.float32)
    qt = quantize_fp8_block(w, 128)
    wd = dequantize_fp8_block(qt)
    assert wd.shape == w.shape


def test_int4_group_roundtrip():
    rng = np.random.default_rng(3)
    w = rng.standard_normal((64, 256)).astype(np.float32)
    qt = quantize_int4_group(w, 128)
    wd = dequantize_int4_group(qt)
    assert wd.shape == w.shape
    # max error bounded by half a quantization step per group
    step = (w.reshape(64, 2, 128).max(-1) - w.reshape(64, 2, 128).min(-1)) / 15.0
    assert (np.abs(wd - w).reshape(64, 2, 128).max(-1) <= step * 0.5 + 1e-6).all()
