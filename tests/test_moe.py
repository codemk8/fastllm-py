"""MoELayer correctness vs a naive dense reference (synthetic weights)."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from fastllm_py.config import ModelConfig
from fastllm_py.expert_cache import GpuExpertCache
from fastllm_py.expert_router import ExpertPlacement, SpeedEstimator, route_topk
from fastllm_py.moe import MoELayer

HID, FF, E, K, T = 64, 96, 8, 2, 17


def make_cfg(**over):
    base = dict(
        model_type="test", num_layers=1, hidden_dim=HID, num_heads=4,
        num_kv_heads=4, head_dim=16, intermediate_dim=FF, vocab_size=100,
        rope_theta=1e4, norm_eps=1e-6, max_position_embeddings=1024,
        tie_word_embeddings=True, attention_bias=False,
        num_experts=E, num_experts_per_tok=K,
    )
    base.update(over)
    return ModelConfig(**base)


def naive_moe(x, gate_w, experts, cfg, e_bias=None):
    """Straightforward per-token reference."""
    logits = x @ gate_w.T
    if cfg.scoring_func == "sigmoid":
        probs = 1 / (1 + np.exp(-logits))
        select = probs + (e_bias if e_bias is not None else 0)
    else:
        ex = np.exp(logits - logits.max(-1, keepdims=True))
        probs = ex / ex.sum(-1, keepdims=True)
        select = probs
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        top = np.argsort(-select[t])[:cfg.num_experts_per_tok]
        w = probs[t, top]
        if cfg.norm_topk_prob:
            w = w / w.sum()
        w = w * cfg.routed_scaling_factor
        for eid, wi in zip(top, w):
            g = x[t] @ experts[eid]["gate"].T.astype(np.float32)
            u = x[t] @ experts[eid]["up"].T.astype(np.float32)
            act = g / (1 + np.exp(-g)) * u
            out[t] += wi * (act @ experts[eid]["down"].T.astype(np.float32))
    return out


@pytest.fixture(scope="module")
def weights():
    rng = np.random.default_rng(42)
    gate_w = rng.standard_normal((E, HID)).astype(np.float32) * 0.5
    experts = {
        e: {p: rng.standard_normal((FF, HID) if p != "down" else (HID, FF))
                 .astype(np.float16) * 0.1
            for p in ("gate", "up", "down")}
        for e in range(E)
    }
    x = rng.standard_normal((T, HID)).astype(np.float32)
    return gate_w, experts, x


def _run(cfg, weights, placement_map, threshold):
    gate_w, experts, x = weights
    layer = MoELayer(
        cfg, 0, cp.asarray(gate_w), experts,
        ExpertPlacement(placement_map), GpuExpertCache(1 << 28),
        SpeedEstimator(threshold=threshold),
    )
    out = cp.asnumpy(layer.forward(cp.asarray(x)))
    ref = naive_moe(x, gate_w, experts, cfg)
    np.testing.assert_allclose(out, ref, rtol=2e-3, atol=2e-3)


def test_all_cpu(weights):
    _run(make_cfg(), weights, {e: "cpu" for e in range(E)}, threshold=10**9)


def test_all_gpu(weights):
    _run(make_cfg(), weights, {e: "cuda:0" for e in range(E)}, threshold=1)


def test_hybrid_split(weights):
    place = {e: ("cuda:0" if e % 2 else "cpu") for e in range(E)}
    _run(make_cfg(), weights, place, threshold=3)


def test_norm_topk(weights):
    _run(make_cfg(norm_topk_prob=True), weights,
         {e: "cpu" for e in range(E)}, threshold=10**9)


def test_sigmoid_scoring_deepseek_style(weights):
    _run(make_cfg(scoring_func="sigmoid", routed_scaling_factor=2.5,
                  norm_topk_prob=True), weights,
         {e: "cpu" for e in range(E)}, threshold=4)


def test_route_topk_task_lists():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((5, E)).astype(np.float32)
    tasks = route_topk(scores, K)
    assert sum(len(t.token_idx) for t in tasks) == 5 * K
    for t in tasks:
        assert len(np.unique(t.token_idx)) == len(t.token_idx)
