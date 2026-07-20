"""Correctness of the speculative-sampling accept/reject step (pure numpy).

The guarantee: for ANY draft distribution q and target distribution p, the token
committed by one draft-draw + rejection_step is distributed exactly as p. This
is what makes sampling-mode speculative decoding quality-neutral vs sampling the
target directly. Verified statistically here, independent of any model.
"""
import numpy as np

from fastllm_py.speculative import rejection_step


def _emit_one(p, q, rng):
    """One speculative token: draw x~q, then accept/correct against p."""
    x = int(rng.choice(p.size, p=q))
    accepted, corr = rejection_step(p, q, x, rng)
    return x if accepted else corr


def _empirical(p, q, n=60000, seed=0):
    rng = np.random.default_rng(seed)
    counts = np.zeros(p.size)
    for _ in range(n):
        counts[_emit_one(p, q, rng)] += 1
    return counts / n


def test_matches_target_when_draft_differs():
    # deliberately mismatched draft vs target
    p = np.array([0.1, 0.2, 0.3, 0.4])
    q = np.array([0.4, 0.3, 0.2, 0.1])
    assert np.allclose(_empirical(p, q), p, atol=0.01)


def test_matches_target_when_draft_equals_target():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    emp = _empirical(p, p.copy())
    assert np.allclose(emp, p, atol=0.01)


def test_matches_target_with_disjoint_support():
    # draft puts mass where target has little, and vice versa
    p = np.array([0.05, 0.05, 0.45, 0.45])
    q = np.array([0.45, 0.45, 0.05, 0.05])
    assert np.allclose(_empirical(p, q), p, atol=0.012)


def test_peaked_target_uniform_draft():
    p = np.array([0.7, 0.1, 0.1, 0.1])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.allclose(_empirical(p, q), p, atol=0.01)


def test_accept_certain_when_p_ge_q_at_x():
    # if p(x) >= q(x), acceptance probability is 1 (never rejects at x)
    rng = np.random.default_rng(3)
    p = np.array([0.6, 0.4])
    q = np.array([0.5, 0.5])
    # x=0: p/q = 1.2 -> always accept
    for _ in range(1000):
        accepted, _ = rejection_step(p, q, 0, rng)
        assert accepted
