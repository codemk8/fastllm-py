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

for mode in ("eager", "graph"):
    bd = BatchedDecoder(m, batch_size=len(prompts), max_len=256)
    outs = bd.generate(prompts, max_new_tokens=N, use_graph=(mode == "graph"))
    ok = all(outs[b] == refs[b] for b in range(len(prompts)))
    print(f"correctness [{mode}] (each seq == single-stream):",
          "MATCH" if ok else "MISMATCH", flush=True)
    if not ok:
        for b in range(len(prompts)):
            if outs[b] != refs[b]:
                print(f"  seq {b}: got {outs[b][:8]} ref {refs[b][:8]}")

# throughput scaling: eager vs graph, aggregate tok/s at various batch sizes
base = prompts[0]
for B in (1, 4, 8, 16):
    line = f"B={B:2d}:"
    for mode in ("eager", "graph"):
        bd = BatchedDecoder(m, batch_size=B, max_len=256)
        if mode == "graph":
            bd.capture()
        bd.prime([base] * B)
        cur = np.array([base[-1]] * B, dtype=np.int64)
        stepf = bd.step_graph if mode == "graph" else bd.step
        stepf(cur); cp.cuda.Device().synchronize()      # warm
        t0 = time.time()
        for _ in range(64):
            cur = stepf(cur).argmax(-1).astype(np.int64)
        cp.cuda.Device().synchronize()
        agg = B * 64 / (time.time() - t0)
        line += f"  {mode} {agg:7.1f} tok/s"
    print(line, flush=True)
print("BATCHED_TEST_DONE")
