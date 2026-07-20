#!/usr/bin/env python
"""Diagnose the MoE decode slowdown: one load, staged measurements.

Stages:
  A. decode 24 tokens after an 8-token prefill  (historically fast)
  B. prefill 128 tokens, decode 24              (historically ~5x slower)
  C. drop expert cache + pool, decode 24 again  (isolates cache state)
Instrumented: time inside get_or_upload (uploads), evictions, pool stats.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def main():
    import cupy as cp

    from fastllm_py import DeviceMap, Model
    from fastllm_py.device_router import MoeDeviceMap
    from fastllm_py.expert_cache import GpuExpertCache

    quant = sys.argv[1] if len(sys.argv) > 1 else "int4"
    model = Model.load(
        "models/Qwen1.5-MoE-A2.7B", DeviceMap({"cuda:0": 1}), dtype="float32",
        moe_device=MoeDeviceMap({"cuda": 1, "cpu": 3}),
        gpu_cache_bytes=8 << 30,
        gpu_expert_quant=quant if quant != "fp16" else "none",
    )
    cache: GpuExpertCache = model._moe_shared["cache"]
    pool = cp.get_default_memory_pool()

    # instrument upload time
    stats = {"upload_s": 0.0, "uploads": 0}
    orig = GpuExpertCache.get_or_upload

    def timed(self, key, cpu_value, stream=None):
        hit = key in self.cache
        t0 = time.perf_counter()
        r = orig(self, key, cpu_value, stream)
        if not hit:
            stats["upload_s"] += time.perf_counter() - t0
            stats["uploads"] += 1
        return r

    GpuExpertCache.get_or_upload = timed

    def decode(n, kvs, last_tok):
        cp.cuda.Device().synchronize()
        stats["upload_s"], stats["uploads"] = 0.0, 0
        m0 = cache.misses
        t0 = time.perf_counter()
        tok = last_tok
        for _ in range(n):
            logits, _ = model.forward(np.asarray([tok]), kvs)
            tok = int(cp.asnumpy(logits[-1]).argmax())
        cp.cuda.Device().synchronize()
        dt = time.perf_counter() - t0
        print(f"    decode: {n/dt:6.2f} tok/s | uploads {stats['uploads']} "
              f"({stats['upload_s']*1e3/n:6.1f} ms/tok) | misses +{cache.misses-m0} "
              f"| hit {cache.hit_rate:.0%} | cache {cache.used/2**30:.2f}GiB "
              f"| pool used {pool.used_bytes()/2**30:.2f} total {pool.total_bytes()/2**30:.2f}GiB "
              f"| graveyard {len(cache._graveyard)}")

    ids8 = np.arange(100, 108, dtype=np.int64)
    print("A: 8-token prefill")
    logits, kvs = model.forward(ids8)
    decode(24, kvs, int(cp.asnumpy(logits[-1]).argmax()))
    decode(24, kvs, 0)

    print("B: 128-token prefill")
    ids128 = np.arange(100, 228, dtype=np.int64)
    t0 = time.perf_counter()
    logits, kvs = model.forward(ids128)
    cp.cuda.Device().synchronize()
    print(f"    prefill: {128/(time.perf_counter()-t0):.1f} tok/s")
    decode(24, kvs, int(cp.asnumpy(logits[-1]).argmax()))
    decode(24, kvs, 0)

    print("C: drop cache + pool, decode again (same 128-tok kv)")
    cache.cache.clear(); cache.lru.clear(); cache.events.clear()
    cache._graveyard.clear(); cache.used = 0
    pool.free_all_blocks()
    decode(24, kvs, 0)
    decode(24, kvs, 0)


if __name__ == "__main__":
    main()
