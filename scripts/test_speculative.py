#!/usr/bin/env python
"""Speculative decoding: correctness (== greedy target) + speedup.
Args: target_path draft_path"""
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
import cupy as cp

from fastllm_py import DeviceMap, Model
from fastllm_py.speculative import SpeculativeDecoder

target_path, draft_path = sys.argv[1], sys.argv[2]
N = 96

t = time.time()
target = Model.load(target_path, DeviceMap({"cuda:0": 1}), linear_quant="int4")
draft = Model.load(draft_path, DeviceMap({"cuda:0": 1}), linear_quant="int4")
print(f"loaded target+draft {time.time()-t:.1f}s "
      f"(vocab {target.cfg.vocab_size}/{draft.cfg.vocab_size})", flush=True)

ids = np.array([785, 6722, 315, 9625, 374], dtype=np.int64)

# reference: pure greedy target
t0 = time.time()
ref = target.generate(ids, max_new_tokens=N)
cp.cuda.Device().synchronize()
ref_s = time.time() - t0

for gamma in (4, 6):
    spec = SpeculativeDecoder(target, draft, gamma=gamma)
    t0 = time.time()
    out = spec.generate(ids, max_new_tokens=N)
    cp.cuda.Device().synchronize()
    spec_s = time.time() - t0
    match = out[:N] == ref[:N]
    acc = spec.stats["accepted"] / max(1, spec.stats["proposed"])
    print(f"gamma={gamma}: {'MATCH' if match else 'MISMATCH'} | "
          f"target fwds {spec.stats['target_forwards']} (vs {N} greedy) | "
          f"accept {acc:.0%} | "
          f"target {ref_s:.2f}s -> spec {spec_s:.2f}s = {ref_s/spec_s:.2f}x", flush=True)
print("SPEC_TEST_DONE")
