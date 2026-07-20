#!/usr/bin/env python
"""deepseek-llm-67b-chat INT4 across 2x4090, eager vs multi-GPU CUDA-graph decode."""
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
import cupy as cp

from fastllm_py import DeviceMap, Model
from fastllm_py.graph_decode import GraphDecoder

PATH = sys.argv[1] if len(sys.argv) > 1 else "/opt/tmp/ypzhang/models/deepseek-llm-67b-chat"
ids = np.array([785, 6722, 315, 9625, 374], dtype=np.int64)

t = time.time()
m = Model.load(PATH, DeviceMap({"cuda:0": 1, "cuda:1": 1}), linear_quant="int4")
print(f"loaded {time.time()-t:.0f}s", flush=True)
for d in (0, 1):
    free, total = cp.cuda.Device(d).mem_info
    print(f"  cuda:{d} used {(total-free)/2**30:.1f}GB", flush=True)

def eager_run(n):
    logits, kvs = m.forward(ids)
    nxt = int(cp.asnumpy(logits[-1]).argmax()); cp.cuda.Device(1).synchronize()
    t0 = time.time()
    for _ in range(n):
        logits, kvs = m.forward(np.asarray([nxt]), kvs)
        nxt = int(cp.asnumpy(logits[-1]).argmax())
    cp.cuda.Device(1).synchronize()
    return n / (time.time() - t0)

gd = GraphDecoder(m)
gd.resize(len(ids) + 64)   # tight max_len: attention is O(max_len)/token
print(f"segments: {len(gd.segments)} "
      f"(layers/dev: {[len(s.layers) for s in gd.segments]}) max_len={gd.max_len}", flush=True)
first, pos = gd.prime(ids)
ok = gd.verify(int(np.argmax(first)), pos)
print(f"graph bit-exact vs eager: {ok}", flush=True)

def graph_run(n):
    nxt = int(np.argmax(first)); p = pos
    t0 = time.time()
    for _ in range(n):
        nxt = int(np.argmax(gd.step(nxt, p))); p += 1
    return n / (time.time() - t0)

e = eager_run(48)
g = graph_run(48)
print(f"decode eager {e:.2f} | graph {g:.2f} tok/s | speedup {g/e:.2f}x", flush=True)
print("RUN_67B_GRAPH_DONE", flush=True)
