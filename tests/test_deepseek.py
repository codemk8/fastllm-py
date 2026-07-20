"""DeepSeek-V2-Lite end-to-end: MLA attention + YaRN + MoE w/ shared experts."""
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get("FASTLLM_TEST_DS_MODEL", str(ROOT / "models" / "DeepSeek-V2-Lite"))
REF_PATH = os.environ.get("FASTLLM_TEST_DS_REF", str(ROOT / "models" / "dsv2lite_ref.npz"))

pytestmark = pytest.mark.skipif(
    not (Path(MODEL_PATH).exists() and Path(REF_PATH).exists()),
    reason="DeepSeek-V2-Lite model or reference not available",
)


@pytest.fixture(scope="module")
def ref():
    return np.load(REF_PATH)


@pytest.fixture(scope="module")
def model():
    from fastllm_py import DeviceMap, Model
    from fastllm_py.device_router import MoeDeviceMap

    return Model.load(
        MODEL_PATH, DeviceMap({"cuda:0": 1}), dtype="float32",
        moe_device=MoeDeviceMap({"cuda": 1, "cpu": 3}),
        expert_dtype="float16", gpu_cache_bytes=6 << 30,
    )


def _np(x):
    import cupy as cp

    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else np.asarray(x)


def test_deepseek_prefill_matches_hf(model, ref):
    logits, _ = model.forward(ref["ids"])
    logits = _np(logits)
    hf = ref["logits"]
    assert logits.shape == hf.shape
    assert (logits.argmax(-1) == hf.argmax(-1)).all()
    err = np.abs(logits[-1] - hf[-1]).max()
    assert err < 0.25, f"max logit err {err}"


def test_deepseek_greedy_generation(model, ref):
    out = model.generate(ref["ids"], max_new_tokens=len(ref["greedy"]))
    assert out == ref["greedy"].tolist(), (out, ref["greedy"].tolist())
