#!/usr/bin/env python
"""Batched decode: correctness (each seq == single-stream) + throughput scaling."""
import sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 2)[0])
import cupy as cp
from fastllm_py import DeviceMap, Model
from fastllm_py.batched import BatchedDecoder

path = sys.argv[1]
m = Model.load(path, DeviceMap({"cuda:0": 1}), linear_quant="int4")

# a few distinct prompts (varied lengths)
prompts = [
    [785, 6722, 315, 9625, 374],            # "The capital of France is"
    [40, 1079, 264, 4128, 1614, 11],        # "I am a language model,"
    [785, 6722, 315, 9625, 374],            # dup of #0 -> must match #0
    [9707, 11, 1879, 0],                    # "Hello, world!"
]
N = 24

# single-stream references
refs = [m.generate(np.asarray(p, dtype=np.int64), max_new_tokens=N) for p in prompts]

bd = BatchedDecoder(m, batch_size=len(prompts), max_len=256)
outs = bd.generate(prompts, max_new_tokens=N)
ok = all(outs[b] == refs[b] for b in range(len(prompts)))
print("correctness (each seq == single-stream):", "MATCH" if ok else "MISMATCH", flush=True)
if not ok:
    for b in range(len(prompts)):
        if outs[b] != refs[b]:
            print(f"  seq {b}: got {outs[b][:8]} ref {refs[b][:8]}")

# throughput scaling: aggregate tok/s at various batch sizes
base = prompts[0]
for B in (1, 4, 8, 16):
    bd = BatchedDecoder(m, batch_size=B, max_len=256)
    pl = [base] * B
    bd.prime(pl)
    cur = np.array([base[-1]] * B, dtype=np.int64)
    bd.step(cur); cp.cuda.Device().synchronize()      # warm
    t0 = time.time()
    for _ in range(64):
        logits = bd.step(cur); cur = logits.argmax(-1).astype(np.int64)
    cp.cuda.Device().synchronize()
    dt = time.time() - t0
    print(f"B={B:2d}: {64/dt:6.1f} steps/s | aggregate {B*64/dt:7.1f} tok/s "
          f"| per-seq {64/dt:6.1f} tok/s", flush=True)
print("BATCHED_TEST_DONE")
