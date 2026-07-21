"""Activation-dtype contract for the graph decoder (pure, no GPU).

Custom RawKernels are compiled for a specific compute dtype and would silently
mis-read memory if handed a different activation dtype (this was a real NaN bug:
the MoE branch makes the residual fp32, and a __half-compiled GEMV read those
fp32 bytes as fp16 pairs). The contract: activations flow only in *registered*
dtypes, kernels are keyed by _act_ctype(), and every kernel input is coerced via
_act_in(). These tests pin the registry + the fail-loud behavior so a new
precision is added deliberately (one _ACT_CTYPE entry), not by accident.
"""
import numpy as np
import pytest

from fastllm_py.graph_decode import _ACT_CTYPE, _act_ctype


def test_registered_dtypes_map_to_ctypes():
    assert _act_ctype(np.float16) == "__half"
    assert _act_ctype(np.float32) == "float"
    # accepts dtype objects and instances alike
    assert _act_ctype(np.dtype("float16")) == "__half"


def test_unregistered_dtype_raises_not_silent():
    # the whole point: an unhandled dtype must fail loudly, never be reinterpreted
    for bad in (np.int8, np.uint8, np.int32, np.float64):
        with pytest.raises(TypeError):
            _act_ctype(bad)


def test_registry_is_the_single_extension_point():
    # adding a precision = one entry here; keep the known ones present
    assert {"float16", "float32"} <= set(_ACT_CTYPE)
    # values are CUDA C scalar type spellings
    assert all(isinstance(v, str) and v for v in _ACT_CTYPE.values())
