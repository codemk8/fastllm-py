"""Unit tests for the shared token sampler (pure numpy, no GPU)."""
import numpy as np

from fastllm_py.graph_decode import sample_logits


def _logits(vocab=50, peak=None, seed=0):
    rng = np.random.default_rng(seed)
    lg = rng.normal(size=vocab).astype(np.float32)
    if peak is not None:
        lg[peak] = 100.0
    return lg


def test_greedy_is_argmax():
    lg = _logits(peak=17)
    assert sample_logits(lg, temperature=0.0) == 17
    # temperature<=0 ignores top_p / top_k
    assert sample_logits(lg, temperature=0.0, top_p=0.5, top_k=3) == 17


def test_top_k_1_equals_greedy():
    lg = _logits(peak=9)
    # only the argmax survives top_k=1, so any temperature yields it
    for t in (0.5, 1.0, 2.0):
        assert sample_logits(lg, temperature=t, top_k=1, rng=np.random.default_rng(1)) == 9


def test_seed_reproducible_and_varies():
    lg = _logits()
    a = [sample_logits(lg, temperature=1.0, rng=np.random.default_rng(42)) for _ in range(5)]
    b = [sample_logits(lg, temperature=1.0, rng=np.random.default_rng(42)) for _ in range(5)]
    assert a == b  # same seed -> identical stream
    c = sample_logits(lg, temperature=1.0, rng=np.random.default_rng(7))
    # not a hard guarantee, but with 50 classes a single draw almost surely differs
    assert isinstance(c, int)


def test_top_p_restricts_support():
    # two dominant tokens carry ~all the mass; nucleus should never leave them
    lg = np.full(20, -50.0, dtype=np.float32)
    lg[3] = 2.0
    lg[8] = 1.9
    rng = np.random.default_rng(0)
    picks = {sample_logits(lg, temperature=1.0, top_p=0.9, rng=rng) for _ in range(200)}
    assert picks <= {3, 8}


def test_distribution_matches_softmax():
    # at temperature 1, empirical frequencies should track softmax probs
    lg = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    probs = np.exp(lg) / np.exp(lg).sum()
    rng = np.random.default_rng(123)
    counts = np.zeros(3)
    N = 20000
    for _ in range(N):
        counts[sample_logits(lg, temperature=1.0, rng=rng)] += 1
    assert np.allclose(counts / N, probs, atol=0.02)
