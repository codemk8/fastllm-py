#!/usr/bin/env python
"""Continuous batching: each request's output == single-stream, + throughput."""
import sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 2)[0])
import cupy as cp
from fastllm_py import DeviceMap, Model
from fastllm_py.batched import BatchedEngine

path = sys.argv[1]
m = Model.load(path, DeviceMap({"cuda:0": 1}), linear_quant="int4")

# heterogeneous requests: different prompts AND different max_new_tokens
reqs = [
    ([785, 6722, 315, 9625, 374], 20),
    ([40, 1079, 264, 4128, 1614, 11], 12),
    ([9707, 11, 1879, 0], 28),
    ([785, 6722, 315, 9625, 374], 16),      # dup prompt, different length
    ([16, 17, 18, 19, 20], 24),
    ([9906, 1917], 18),
]

# single-stream references
refs = [m.generate(np.asarray(p, dtype=np.int64), max_new_tokens=n) for p, n in reqs]

# continuous batching: max_batch smaller than #requests -> forces slot churn
eng = BatchedEngine(m, max_batch=4, max_len=256)
collected = [[] for _ in reqs]
for i, (p, n) in enumerate(reqs):
    eng.submit(p, max_new_tokens=n, on_token=(lambda t, i=i: collected[i].append(t)))
eng.run_until_idle()

ok = all(collected[i] == refs[i] for i in range(len(reqs)))
print("continuous batching correctness (each req == single-stream):",
      "MATCH" if ok else "MISMATCH", flush=True)
if not ok:
    for i in range(len(reqs)):
        if collected[i] != refs[i]:
            print(f"  req {i}: got {collected[i][:8]} ref {refs[i][:8]}")

# sustained throughput: flood N requests, max_batch B, fixed gen length
for B in (8, 16):
    eng = BatchedEngine(m, max_batch=B, max_len=256)
    Nreq, gen = 64, 32
    done = [0]
    for _ in range(Nreq):
        eng.submit([785, 6722, 315, 9625, 374], max_new_tokens=gen,
                   on_token=lambda t: done.__setitem__(0, done[0] + 1))
    cp.cuda.Device().synchronize()
    t0 = time.time()
    eng.run_until_idle()
    cp.cuda.Device().synchronize()
    dt = time.time() - t0
    print(f"max_batch={B:2d}: {Nreq} reqs x {gen} tok = {done[0]} tokens in "
          f"{dt:.2f}s = {done[0]/dt:7.1f} tok/s", flush=True)
print("CONTINUOUS_TEST_DONE")
