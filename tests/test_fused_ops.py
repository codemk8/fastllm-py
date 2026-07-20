"""Fused GPU RMSNorm kernel == numpy reference (activated after verification)."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from fastllm_py.kernels import ops


def _ref_rmsnorm(x, w, eps):
    xf = x.astype(np.float32)
    var = np.mean(xf * xf, axis=-1, keepdims=True)
    return ((xf / np.sqrt(var + eps)) * w.astype(np.float32)).astype(x.dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("shape", [(5, 1024), (1, 2048), (17, 4, 128)])
def test_fused_rmsnorm_matches_reference(dtype, shape):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(dtype)
    w = rng.standard_normal(shape[-1]).astype(dtype)
    eps = 1e-6
    ref = _ref_rmsnorm(x, w, eps)
    got = cp.asnumpy(ops._rmsnorm_cupy(cp.asarray(x), cp.asarray(w), eps))
    atol = 2e-3 if dtype == np.float16 else 1e-5
    np.testing.assert_allclose(got.astype(np.float32), ref.astype(np.float32),
                               rtol=1e-3, atol=atol)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_fused_swiglu_matches_reference(dtype):
    rng = np.random.default_rng(1)
    g = rng.standard_normal((7, 256)).astype(dtype)
    u = rng.standard_normal((7, 256)).astype(dtype)
    gf = g.astype(np.float32)
    ref = ((gf / (1.0 + np.exp(-gf))) * u.astype(np.float32)).astype(dtype)
    got = cp.asnumpy(ops._swiglu_cupy(cp.asarray(g), cp.asarray(u)))
    atol = 2e-3 if dtype == np.float16 else 1e-5
    np.testing.assert_allclose(got.astype(np.float32), ref.astype(np.float32),
                               rtol=1e-3, atol=atol)
