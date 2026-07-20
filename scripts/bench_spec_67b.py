#!/usr/bin/env python
"""Speculative decode on the 67B: deepseek-llm-67b (target, 2 GPU) +
deepseek-llm-7b (draft, 1 GPU). The 67B is bandwidth-bound so graph decode only
gets ~1.34x; speculative is the real lever. Measures decode tok/s + acceptance
across gamma, vs the target's own graph/eager decode. Same-family models share
the 102400 vocab, so this is a valid greedy draft."""
import sys, time
sys.path.insert(0, __file__.rsplit("/", 2)[0])
import numpy as np, cupy as cp
from transformers import AutoTokenizer
from fastllm_py import DeviceMap, Model
from fastllm_py.speculative import SpeculativeDecoder

TGT = "/opt/tmp/ypzhang/models/deepseek-llm-67b-chat"
DRF = "/opt/tmp/ypzhang/models/deepseek-llm-7b-chat"

tok = AutoTokenizer.from_pretrained(TGT)
t = time.time()
target = Model.load(TGT, DeviceMap({"cuda:0": 1, "cuda:1": 1}), linear_quant="int4")
print(f"[target 67B INT4 2-GPU loaded {time.time()-t:.0f}s]", flush=True)
t = time.time()
draft = Model.load(DRF, DeviceMap({"cuda:1": 1}), linear_quant="int4")
print(f"[draft 7B INT4 on cuda:1 loaded {time.time()-t:.0f}s]", flush=True)

prompt = "Explain how a rocket reaches orbit, step by step."
ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                              add_generation_prompt=True, tokenize=True)
if hasattr(ids, "input_ids"):
    ids = ids.input_ids
ids = np.asarray(ids, dtype=np.int64)
N = 128

# --- baseline: target greedy, graph fast path (auto) vs forced eager ---
def target_decode(use_graph, n=N):
    t0 = time.time()
    out = target.generate(ids, max_new_tokens=n, use_graph=use_graph)
    cp.cuda.Device(0).synchronize()
    return out, n / (time.time() - t0)

g_out, g_tps = target_decode(True)
print(f"target graph : {g_tps:5.1f} tok/s", flush=True)
e_out, e_tps = target_decode(False)
print(f"target eager : {e_tps:5.1f} tok/s", flush=True)
assert g_out == e_out, "graph != eager on target!"

# --- speculative across gamma ---
for gamma in (2, 4, 6):
    spec = SpeculativeDecoder(target, draft, gamma=gamma)
    t0 = time.time()
    s_out = spec.generate(ids, max_new_tokens=N)
    cp.cuda.Device(0).synchronize()
    tps = N / (time.time() - t0)
    s = spec.stats
    acc = s["accepted"] / s["proposed"] if s["proposed"] else 0
    match = "OK" if s_out[:len(g_out)] == g_out[:len(s_out)] else "MISMATCH"
    print(f"spec gamma={gamma}: {tps:5.1f} tok/s | accept {acc:4.0%} | "
          f"target fwds {s['target_forwards']:3d} (vs {N}) | greedy-match {match} | "
          f"{tps/g_tps:.2f}x graph", flush=True)

print("SPEC_67B_DONE")
