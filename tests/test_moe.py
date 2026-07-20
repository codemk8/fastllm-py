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


def test_marlin_int4_gpu_experts(weights):
    from fastllm_py.kernels import marlin

    if not marlin.available():
        pytest.skip("native marlin .so not built")
    gate_w, experts, x = weights
    # marlin needs dims %64: synthesize larger experts
    rng = np.random.default_rng(7)
    HID2, FF2 = 128, 256
    experts2 = {
        e: {p: (rng.standard_normal((FF2, HID2) if p != "down" else (HID2, FF2))
                * 0.1).astype(np.float16)
            for p in ("gate", "up", "down")}
        for e in range(E)
    }
    gate_w2 = rng.standard_normal((E, HID2)).astype(np.float32) * 0.5
    x2 = rng.standard_normal((T, HID2)).astype(np.float32)
    cfg = make_cfg(hidden_dim=HID2, intermediate_dim=FF2)

    from fastllm_py.moe import build_marlin_expert_payload

    payloads = {e: build_marlin_expert_payload(experts2[e]) for e in range(E)}
    layer = MoELayer(
        cfg, 0, cp.asarray(gate_w2), experts2,
        ExpertPlacement({e: "cuda:0" for e in range(E)}),
        GpuExpertCache(1 << 28), SpeedEstimator(threshold=1),
        gpu_payloads=payloads,
    )
    out = cp.asnumpy(layer.forward(cp.asarray(x2)))

    # reference must use the DEQUANTIZED weights: this isolates the marlin
    # plumbing from int4 quantization noise (which is ~13%/proj on gaussians)
    def dequant(w, gs=128):
        n, k = w.shape
        g = w.astype(np.float32).reshape(n, k // gs, gs)
        wmin, wmax = g.min(axis=2), g.max(axis=2)
        scale = np.where(wmax - wmin == 0, 1.0, (wmax - wmin) / 15.0).astype(np.float32)
        zero = np.clip(np.rint(-wmin / scale), 0, 15)
        q = np.clip(np.rint(g / scale[:, :, None]) + zero[:, :, None], 0, 15)
        return ((q - zero[:, :, None]) * scale[:, :, None]).reshape(n, k).astype(np.float16)

    experts_dq = {e: {p: dequant(w) for p, w in ws.items()}
                  for e, ws in experts2.items()}
    ref = naive_moe(x2, gate_w2, experts_dq, cfg)
    rel = np.abs(out - ref).mean() / (np.abs(ref).mean() + 1e-9)
    assert rel < 0.02, rel


def test_eviction_under_pressure(weights):
    """Cache sized for ~2 experts: every forward evicts. Deferred eviction
    must stay correct with no device-wide syncs."""
    gate_w, experts, x = weights
    per_expert = sum(v.nbytes for v in experts[0].values())
    cache = GpuExpertCache(int(per_expert * 2.5))
    layer = MoELayer(
        make_cfg(), 0, cp.asarray(gate_w), experts,
        ExpertPlacement({e: "cuda:0" for e in range(E)}), cache,
        SpeedEstimator(threshold=1),
    )
    ref = naive_moe(x, gate_w, experts, make_cfg())
    for _ in range(6):
        out = cp.asnumpy(layer.forward(cp.asarray(x)))
        np.testing.assert_allclose(out, ref, rtol=2e-3, atol=2e-3)
    assert cache.misses > cache.hits  # eviction actually happened


def test_route_topk_task_lists():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((5, E)).astype(np.float32)
    tasks = route_topk(scores, K)
    assert sum(len(t.token_idx) for t in tasks) == 5 * K
    for t in tasks:
        assert len(np.unique(t.token_idx)) == len(t.token_idx)
