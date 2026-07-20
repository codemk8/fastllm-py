"""Group-limited (DeepSeek V3/V4 node-limited) expert routing."""
import numpy as np

from fastllm_py.expert_router import route_topk


def _selected(scores, top_k, **kw):
    """route_topk -> dict token -> set(expert ids)."""
    tasks = route_topk(scores, top_k, **kw)
    sel = {}
    for t in tasks:
        for tok in t.token_idx.tolist():
            sel.setdefault(int(tok), set()).add(t.expert_id)
    return sel


def _ref_group(logits, top_k, n_group, topk_group, bias):
    """Straightforward reference for V3 group routing."""
    T, E = logits.shape
    gsz = E // n_group
    probs = 1.0 / (1.0 + np.exp(-logits))
    select = probs + bias
    gs = select.reshape(T, n_group, gsz)
    group_score = np.sort(gs, -1)[:, :, -2:].sum(-1)
    out = {}
    for t in range(T):
        keep = set(np.argsort(-group_score[t])[:topk_group].tolist())
        masked = select[t].copy()
        for g in range(n_group):
            if g not in keep:
                masked[g * gsz:(g + 1) * gsz] = -np.inf
        out[t] = set(np.argsort(-masked)[:top_k].tolist())
    return out


def test_group_routing_only_selects_from_top_groups():
    # 4 groups of 2; groups 0 and 2 clearly strongest -> experts {0,1,4,5} only
    logits = np.array([[5, 4, -9, -9, 6, 3, -9, -9]], dtype=np.float32)
    bias = np.zeros(8, dtype=np.float32)
    sel = _selected(logits, top_k=2, scoring="sigmoid", e_score_bias=bias,
                    n_group=4, topk_group=2)
    assert sel[0] <= {0, 1, 4, 5}          # never an expert from a dropped group
    assert sel[0] == {0, 4}                # the two highest within kept groups


def test_group_routing_matches_reference_random():
    rng = np.random.default_rng(0)
    E, n_group, topk_group, top_k, T = 16, 4, 2, 4, 6
    logits = rng.standard_normal((T, E)).astype(np.float32)
    bias = (rng.standard_normal(E) * 0.1).astype(np.float32)
    got = _selected(logits, top_k, scoring="sigmoid", e_score_bias=bias,
                    n_group=n_group, topk_group=topk_group)
    ref = _ref_group(logits, top_k, n_group, topk_group, bias)
    for t in range(T):
        assert got[t] == ref[t], (t, got[t], ref[t])


def test_no_groups_is_unchanged():
    # n_group=0 -> plain top-k (regression guard for Qwen/Mixtral MoE)
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((4, 8)).astype(np.float32)
    a = _selected(logits, 2, scoring="softmax")
    b = _selected(logits, 2, scoring="softmax", n_group=0, topk_group=0)
    assert a == b
