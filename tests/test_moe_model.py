"""End-to-end hybrid MoE forward on Qwen1.5-MoE-A2.7B vs saved HF reference.

Reference is produced by scripts/make_reference.py (fp32 CPU HF forward).
Our engine: attention/shared-experts fp32 on GPU, routed experts fp16 in RAM
with hybrid CPU/GPU execution — so tolerances reflect fp16 expert storage.
"""
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get("FASTLLM_TEST_MOE_MODEL", str(ROOT / "models" / "Qwen1.5-MoE-A2.7B"))
REF_PATH = os.environ.get("FASTLLM_TEST_MOE_REF", str(ROOT / "models" / "qwen15moe_ref.npz"))

pytestmark = pytest.mark.skipif(
    not (Path(MODEL_PATH).exists() and Path(REF_PATH).exists()),
    reason="MoE model or reference not available",
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
        moe_device=MoeDeviceMap({"cuda": 1, "cpu": 3}),  # 25% experts GPU-resident
        expert_dtype="float16", gpu_cache_bytes=6 << 30,
    )


def _np(x):
    import cupy as cp

    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else np.asarray(x)


def test_moe_prefill_matches_hf(model, ref):
    logits, _ = model.forward(ref["ids"])
    logits = _np(logits)
    hf = ref["logits"]
    assert logits.shape == hf.shape
    assert (logits.argmax(-1) == hf.argmax(-1)).all()
    # fp16 expert storage: modest absolute tolerance on final logits
    err = np.abs(logits[-1] - hf[-1]).max()
    assert err < 0.25, f"max logit err {err}"
    cos = (logits[-1] @ hf[-1]) / (np.linalg.norm(logits[-1]) * np.linalg.norm(hf[-1]))
    assert cos > 0.9999, cos


def test_moe_greedy_generation(model, ref):
    out = model.generate(ref["ids"], max_new_tokens=len(ref["greedy"]))
    assert out == ref["greedy"].tolist(), (out, ref["greedy"].tolist())


def test_expert_cache_stats(model, ref):
    cache = model._moe_shared["cache"]
    total = cache.hits + cache.misses
    assert total > 0
    print(f"expert cache: hit_rate={cache.hit_rate:.2%} used={cache.used/2**30:.2f} GiB")
