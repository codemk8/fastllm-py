"""KVCache capacity-doubling buffer == naive concatenate (CPU, no GPU)."""
import numpy as np

from fastllm_py.model import KVCache


def test_growth_matches_concatenate():
    rng = np.random.default_rng(0)
    kv = KVCache()
    ref_k, ref_v = None, None
    # prefill 5, then 20 single-token decode steps -> forces several regrows
    for step, n in enumerate([5] + [1] * 20):
        k_new = rng.standard_normal((n, 4, 8)).astype(np.float32)
        v_new = rng.standard_normal((n, 4, 8)).astype(np.float32)
        k_all, v_all = kv.append(k_new, v_new, np)
        ref_k = k_new if ref_k is None else np.concatenate([ref_k, k_new], 0)
        ref_v = v_new if ref_v is None else np.concatenate([ref_v, v_new], 0)
        np.testing.assert_array_equal(k_all, ref_k)
        np.testing.assert_array_equal(v_all, ref_v)
        assert kv.seq_len == ref_k.shape[0]


def test_empty_cache():
    kv = KVCache()
    assert kv.seq_len == 0
    assert kv.k is None and kv.v is None
