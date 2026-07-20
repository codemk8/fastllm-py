#!/usr/bin/env python
"""Validate + benchmark multi-segment (multi-GPU) graph decode.
Arg: model path. Tests DeviceMap on 1 GPU then split across 2 GPUs."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
import cupy as cp

from fastllm_py import DeviceMap, Model
from fastllm_py.graph_decode import GraphDecoder

path = sys.argv[1]
N = 24
ids = np.array([785, 6722, 315, 9625, 374], dtype=np.int64)

for label, devmap in [("1-GPU", {"cuda:0": 1}), ("2-GPU", {"cuda:0": 1, "cuda:1": 1})]:
    print(f"=== {label} {devmap} ===", flush=True)
    t = time.time()
    m = Model.load(path, DeviceMap(devmap), linear_quant="int4")
    ndev = len({l.device for l in m.layers})
    print(f"loaded {time.time()-t:.1f}s, {ndev} device(s)", flush=True)

    ref = m.generate(ids, max_new_tokens=N)
    gd = GraphDecoder(m, max_len=256)
    nseg = len(gd.segments)
    out = gd.generate(ids, max_new_tokens=N, use_graph=True)
    mode = "eager-fallback" if gd.graph_fellback else "graph"
    print(f"segments={nseg}  {'MATCH' if out == ref else 'MISMATCH'} [{mode}]", flush=True)

    # decode speed: eager model vs graph
    def eager_run(n):
        logits, kvs = m.forward(ids)
        nxt = int(cp.asnumpy(logits[-1]).argmax())
        cp.cuda.Device().synchronize()
        t0 = time.time()
        for _ in range(n):
            logits, kvs = m.forward(np.asarray([nxt]), kvs)
            nxt = int(cp.asnumpy(logits[-1]).argmax())
        cp.cuda.Device().synchronize()
        return n / (time.time() - t0)

    gd.capture()
    first, pos = gd.prime(ids)
    def graph_run(n):
        nxt = int(np.argmax(first)); p = pos
        t0 = time.time()
        for _ in range(n):
            nxt = int(np.argmax(gd.step(nxt, p))); p += 1
        return n / (time.time() - t0)

    e, g = eager_run(64), graph_run(64)
    print(f"decode eager {e:.1f} | graph {g:.1f} tok/s | speedup {g/e:.2f}x", flush=True)
    del m, gd
    cp.get_default_memory_pool().free_all_blocks()
print("MULTIGPU_TEST_DONE")
