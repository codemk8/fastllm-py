"""End-to-end forward pass vs HuggingFace reference (Qwen3-0.6B)."""
import os
from pathlib import Path

import numpy as np
import pytest

MODEL_PATH = os.environ.get(
    "FASTLLM_TEST_MODEL",
    str(Path(__file__).resolve().parent.parent / "models" / "Qwen3-0.6B"),
)

pytestmark = pytest.mark.skipif(
    not Path(MODEL_PATH).exists(), reason=f"model not downloaded at {MODEL_PATH}"
)

PROMPT_IDS = None  # filled by fixture


@pytest.fixture(scope="module")
def token_ids():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    return np.asarray(tok("The capital of France is").input_ids, dtype=np.int64)


@pytest.fixture(scope="module")
def hf_logits(token_ids):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(token_ids)[None])
    return out.logits[0].numpy()


@pytest.fixture(scope="module")
def our_model():
    from fastllm_py import DeviceMap, Model

    return Model.load(MODEL_PATH, DeviceMap({"cuda:0": 1}), dtype="float32")


def _as_numpy(x):
    try:
        import cupy as cp

        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except ImportError:
        pass
    return np.asarray(x)


def test_prefill_logits_match_hf(our_model, token_ids, hf_logits):
    logits, _ = our_model.forward(token_ids)
    logits = _as_numpy(logits)
    assert logits.shape == hf_logits.shape
    # same argmax on every position, tight numeric agreement on the last
    assert (logits.argmax(-1) == hf_logits.argmax(-1)).all()
    np.testing.assert_allclose(logits[-1], hf_logits[-1], rtol=1e-3, atol=2e-2)


def test_decode_matches_prefill(our_model, token_ids):
    """Incremental decode with KV cache == one-shot prefill."""
    full, _ = our_model.forward(token_ids)
    full = _as_numpy(full)

    logits, kvs = our_model.forward(token_ids[:-1])
    step, kvs = our_model.forward(token_ids[-1:], kvs)
    step = _as_numpy(step)
    np.testing.assert_allclose(step[0], full[-1], rtol=1e-4, atol=1e-3)


def test_greedy_generation_matches_hf(our_model, token_ids):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        ref = model.generate(torch.tensor(token_ids)[None], max_new_tokens=16,
                             do_sample=False, temperature=None, top_p=None, top_k=None)
    ref_new = ref[0, len(token_ids):].tolist()
    ours = our_model.generate(token_ids, max_new_tokens=16)
    assert ours == ref_new, (ours, ref_new)
