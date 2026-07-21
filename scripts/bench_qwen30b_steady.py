#!/usr/bin/env python
"""Steady-state decode benchmark for Qwen3-30B-A3B on the 5090 — measures ONLY
the per-token graph replay (prime once, then time N step() calls), excluding
prefill + capture. Also prints an eager-vs-graph steady comparison and the
first-token verify. This corrects bench_qwen30b_5090.py which folded prefill +
resize/recapture into the per-token number."""
import sys, time
sys.path.insert(0, __file__.rsplit("/", 2)[0])
import numpy as np, cupy as cp
from transformers import AutoTokenizer
from fastllm_py import DeviceMap, Model
from fastllm_py.device_router import MoeDeviceMap
from fastllm_py.graph_decode import GraphDecoder, graph_capable

path = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/models/Qwen3-30B-A3B"
tok = AutoTokenizer.from_pretrained(path)
t = time.time()
m = Model.load(path, DeviceMap({"cuda:0": 1}), linear_quant="int4",
               moe_device=MoeDeviceMap({"cuda": 1}), gpu_expert_quant="int4",
               gpu_cache_bytes=22 << 30)
print(f"[loaded {time.time()-t:.0f}s | graph_capable={graph_capable(m)}]", flush=True)

ids = np.array([785, 6722, 315, 9625, 374, 264, 2421, 315], dtype=np.int64)

gd = GraphDecoder(m, max_len=256)
t = time.time(); gd.capture(); print(f"[capture {time.time()-t:.1f}s]", flush=True)
first, pos = gd.prime(ids)

# steady-state graph decode: time only step() replays
def steady(step_fn, n, p0):
    nx = int(np.argmax(first)); p = p0
    step_fn(nx, p)  # warm
    cp.cuda.Device(0).synchronize(); t0 = time.time()
    for _ in range(n):
        nx = int(np.argmax(step_fn(nx, p))); p += 1
    cp.cuda.Device(0).synchronize()
    return n / (time.time() - t0)

g = steady(gd.step, 200, pos)
first, pos = gd.prime(ids)
e = steady(gd.step_eager, 60, pos)
print(f"STEADY decode: eager {e:5.1f} tok/s | graph {g:5.1f} tok/s | {g/e:.2f}x", flush=True)
print(f"  graph ms/token: {1000/g:.2f} ms", flush=True)
print(f"vs ktransformers 52.5 tok/s (their box): {g/52.5:.2f}x", flush=True)

# correctness: graph (indexed-MoE fast path) vs marlin reference (use_graph=False)
ref = m.generate(ids, max_new_tokens=16, use_graph=False)
gph = m.generate(ids, max_new_tokens=16, use_graph=True)
print(f"greedy graph==marlin-ref: {gph == ref}", flush=True)
if gph != ref:
    print("  ref :", ref, "\n  graph:", gph, flush=True)
print("STEADY_DONE", flush=True)
