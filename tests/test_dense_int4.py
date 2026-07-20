"""Dense Marlin-INT4 path: quantized model vs fp32 model on Qwen3-0.6B."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get("FASTLLM_TEST_MODEL", str(ROOT / "models" / "Qwen3-0.6B"))

pytestmark = pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="model missing")


@pytest.fixture(scope="module")
def marlin_ok():
    from fastllm_py.kernels import marlin

    if not marlin.available():
        pytest.skip("native marlin .so not built")


def test_dense_int4_close_to_fp32(marlin_ok):
    import cupy as cp

    from fastllm_py import DeviceMap, Model

    ids = np.array([785, 6722, 315, 9625, 374], dtype=np.int64)  # "The capital of France is"
    ref_model = Model.load(MODEL_PATH, DeviceMap({"cuda:0": 1}), dtype="float32")
    ref, _ = ref_model.forward(ids)
    ref = cp.asnumpy(ref)
    del ref_model
    cp.get_default_memory_pool().free_all_blocks()

    q_model = Model.load(MODEL_PATH, DeviceMap({"cuda:0": 1}), linear_quant="int4")
    out, _ = q_model.forward(ids)
    out = cp.asnumpy(out).astype(np.float32)

    # plain RTN int4 group-128 (no AWQ calibration): 0.6B is the worst case;
    # measured cos ~0.974 with correct greedy continuation
    cos = (out[-1] @ ref[-1]) / (np.linalg.norm(out[-1]) * np.linalg.norm(ref[-1]))
    assert cos > 0.96, cos
    assert out[-1].argmax() == ref[-1].argmax()

    # marlin payload cache round-trip: second load gives identical logits
    assert (Path(MODEL_PATH) / ".marlin_cache").exists()
    q2 = Model.load(MODEL_PATH, DeviceMap({"cuda:0": 1}), linear_quant="int4")
    out2, _ = q2.forward(ids)
    np.testing.assert_array_equal(cp.asnumpy(out2), cp.asnumpy(out).astype(np.float16))


def test_dense_int4_generates_sensible_text(marlin_ok):
    from transformers import AutoTokenizer

    from fastllm_py import DeviceMap, Model

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    ids = np.asarray(tok("The capital of France is").input_ids, dtype=np.int64)
    model = Model.load(MODEL_PATH, DeviceMap({"cuda:0": 1}), linear_quant="int4")
    out = model.generate(ids, max_new_tokens=8)
    text = tok.decode(out)
    assert "Paris" in text, text
