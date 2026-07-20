#!/usr/bin/env python
"""Validate + benchmark CUDA-graph decode vs eager. Arg: model path."""
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

t = time.time()
m = Model.load(path, DeviceMap({"cuda:0": 1}), linear_quant="int4")
print(f"loaded (int4) {time.time()-t:.1f}s", flush=True)

ref = m.generate(ids, max_new_tokens=N)
print("ref    :", ref, flush=True)

# Stage 1: fixed-buffer eager
gd1 = GraphDecoder(m, max_len=256)
s1 = gd1.generate(ids, max_new_tokens=N, use_graph=False)
print("stage1 :", s1, "MATCH" if s1 == ref else "MISMATCH", flush=True)

# Stage 2: CUDA graph (auto-falls-back to eager if not bit-exact)
gd2 = GraphDecoder(m, max_len=256)
s2 = gd2.generate(ids, max_new_tokens=N, use_graph=True)
mode = "eager-fallback" if gd2.graph_fellback else "graph"
print("stage2 :", s2, "MATCH" if s2 == ref else "MISMATCH", f"[{mode}]", flush=True)

# --- decode speed: eager model vs graph ---
def bench(fn, n=64):
    logits, kvs = m.forward(ids)
    nxt = int(cp.asnumpy(logits[-1]).argmax())
    cp.cuda.Device().synchronize()
    t0 = time.time()
    fn(nxt, n)
    cp.cuda.Device().synchronize()
    return n / (time.time() - t0)

def eager_run(nxt, n):
    logits, kvs = m.forward(ids)
    for _ in range(n):
        logits, kvs = m.forward(np.array([nxt]), kvs)
        nxt = int(cp.asnumpy(logits[-1]).argmax())

gd3 = GraphDecoder(m, max_len=256)
gd3.capture()
first, pos = gd3.prime(ids)
def graph_run(nxt, n):
    p = pos
    for _ in range(n):
        lo = gd3.step(nxt, p); p += 1
        nxt = int(np.argmax(lo))

t0 = time.time(); eager_run(0, 64); eager = 64 / (time.time() - t0)
t0 = time.time(); graph_run(int(np.argmax(first)), 64); graph = 64 / (time.time() - t0)
print(f"decode eager: {eager:.1f} tok/s | graph: {graph:.1f} tok/s | speedup {graph/eager:.2f}x",
      flush=True)
print("GRAPH_TEST_DONE", flush=True)
