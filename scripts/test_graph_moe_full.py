#!/usr/bin/env python
"""Graph-capture the whole MoE decode (attention + fused MoE). Validate vs eager
+ benchmark. Arg: MoE model path."""
import sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 2)[0])
import cupy as cp
from fastllm_py import DeviceMap, Model
from fastllm_py.device_router import MoeDeviceMap
from fastllm_py.graph_decode import GraphDecoder, graph_capable

path = sys.argv[1]
t = time.time()
m = Model.load(path, DeviceMap({"cuda:0": 1}), linear_quant="int4",
               moe_device=MoeDeviceMap({"cuda": 1}), gpu_expert_quant="int4",
               gpu_cache_bytes=12 << 30)
print(f"loaded {time.time()-t:.0f}s | graph_capable={graph_capable(m)}", flush=True)
ids = np.array([785, 6722, 315, 9625, 374], dtype=np.int64)
N = 24

ref = m.generate(ids, max_new_tokens=N)
print("ref  :", ref, flush=True)

gd = GraphDecoder(m, max_len=128)
out = gd.generate(ids, max_new_tokens=N, use_graph=True)
mode = "eager-fallback" if gd.graph_fellback else "graph"
print("graph:", out, "MATCH" if out == ref else "MISMATCH", f"[{mode}]", flush=True)

def eager_run(n):
    lg, kv = m.forward(ids); nx = int(cp.asnumpy(lg[-1]).argmax()); cp.cuda.Device().synchronize()
    t0 = time.time()
    for _ in range(n):
        lg, kv = m.forward(np.asarray([nx]), kv); nx = int(cp.asnumpy(lg[-1]).argmax())
    cp.cuda.Device().synchronize(); return n / (time.time() - t0)

gd.capture(); first, pos = gd.prime(ids)
def graph_run(n):
    nx = int(np.argmax(first)); p = pos; t0 = time.time()
    for _ in range(n): nx = int(np.argmax(gd.step(nx, p))); p += 1
    return n / (time.time() - t0)

e, g = eager_run(64), graph_run(64)
print(f"decode eager {e:.1f} | graph {g:.1f} tok/s | speedup {g/e:.2f}x "
      f"(fused-eager baseline was ~33.7)", flush=True)
print("GRAPH_MOE_FULL_DONE")
