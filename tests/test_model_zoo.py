"""Generic correctness spot-checks: any model with a saved HF reference.

References live at models/refs/<model_dir>_ref.npz (scripts/make_reference.py).
Each entry: model dir name + engine load overrides. Skips models/refs not
present, so the suite grows as downloads/references land.
"""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

ZOO = [
    # (model dir, load kwargs, atol on final logits)
    ("deepseek-coder-1.3b-instruct", {}, 0.05),
    ("DeepSeek-R1-Distill-Qwen-1.5B", {}, 0.05),
    # 7B fp32 doesn't fit one 4090 -> split layers across both GPUs (exact math)
    ("deepseek-llm-7b-chat", {"device": {"cuda:0": 1, "cuda:1": 1}}, 0.05),
    ("deepseek-moe-16b-chat", {"moe": True}, 0.25),  # fp16 experts
]


def _cases():
    for name, kw, atol in ZOO:
        yield pytest.param(name, kw, atol, id=name)


@pytest.mark.parametrize("name,kw,atol", _cases())
def test_zoo_prefill_matches_hf(name, kw, atol):
    model_dir = ROOT / "models" / name
    ref_path = ROOT / "models" / "refs" / f"{name}_ref.npz"
    if not model_dir.exists() or not ref_path.exists():
        pytest.skip(f"{name}: model or reference missing")

    import cupy as cp

    from fastllm_py import DeviceMap, Model
    from fastllm_py.device_router import MoeDeviceMap

    ref = np.load(ref_path)
    model = Model.load(
        str(model_dir),
        DeviceMap(kw.get("device", {"cuda:0": 1})),
        dtype="float32",
        moe_device=MoeDeviceMap({"cuda": 1, "cpu": 3}) if kw.get("moe") else None,
        expert_dtype="float16",
        gpu_cache_bytes=6 << 30,
    )
    logits, _ = model.forward(ref["ids"])
    logits = cp.asnumpy(logits) if not isinstance(logits, np.ndarray) else logits
    hf = ref["logits"]
    assert logits.shape == hf.shape
    assert (logits.argmax(-1) == hf.argmax(-1)).all(), "argmax mismatch"
    err = np.abs(logits[-1] - hf[-1]).max()
    assert err < atol, f"max final-logit err {err}"
