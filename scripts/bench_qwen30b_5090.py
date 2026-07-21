#!/usr/bin/env python
"""Qwen3-30B-A3B INT4 resident decode benchmark on a single RTX 5090 — the
apple-to-apple vs the ktransformers leaderboard (Qwen3-30B-A3B BF16, 1x5090
+AVX2 -> 52.5 tok/s decode). We run INT4 experts GPU-resident with the fused
selective-MoE kernel + CUDA-graph decode. Prints load time, a coherence sample,
and eager vs graph decode tok/s.

Usage: bench_qwen30b_5090.py [model_dir]
"""
import sys, time
sys.path.insert(0, __file__.rsplit("/", 2)[0])
import numpy as np, cupy as cp
from transformers import AutoTokenizer
from fastllm_py import DeviceMap, Model
from fastllm_py.device_router import MoeDeviceMap
from fastllm_py.graph_decode import graph_capable

path = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/models/Qwen3-30B-A3B"
tok = AutoTokenizer.from_pretrained(path)

t = time.time()
m = Model.load(path, DeviceMap({"cuda:0": 1}), linear_quant="int4",
               moe_device=MoeDeviceMap({"cuda": 1}), gpu_expert_quant="int4",
               gpu_cache_bytes=22 << 30)
load_s = time.time() - t
free, total = cp.cuda.Device(0).mem_info
print(f"[loaded Qwen3-30B-A3B INT4 in {load_s:.0f}s | graph_capable={graph_capable(m)} "
      f"| VRAM used {(total-free)/2**30:.1f}/{total/2**30:.0f} GB]", flush=True)

# ---- coherence sample (chat) ----
msg = [{"role": "user", "content": "In one sentence, what makes mixture-of-experts models efficient?"}]
ids = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=True)
if hasattr(ids, "input_ids"):
    ids = ids.input_ids
ids = np.asarray(ids, dtype=np.int64)
out = m.generate(ids, max_new_tokens=64, stop_ids={tok.eos_token_id})
print("SAMPLE:", tok.decode([t for t in out if t != tok.eos_token_id]).strip()[:400], flush=True)

# ---- decode throughput: eager vs graph ----
prompt = np.asarray(ids, dtype=np.int64)
N = 128

def bench(use_graph, n=N):
    m.generate(prompt, max_new_tokens=8, use_graph=use_graph)   # warm/capture
    cp.cuda.Device(0).synchronize()
    t0 = time.time()
    o = m.generate(prompt, max_new_tokens=n, use_graph=use_graph)
    cp.cuda.Device(0).synchronize()
    return o, n / (time.time() - t0)

g_out, g = bench(True)
e_out, e = bench(False)
print(f"decode: eager {e:5.1f} tok/s | graph {g:5.1f} tok/s | graph speedup {g/e:.2f}x",
      flush=True)
print(f"vs ktransformers (Qwen3-30B-A3B BF16 1x5090+AVX2 = 52.5 tok/s): "
      f"graph is {g/52.5:.2f}x their number", flush=True)
print("greedy graph==eager:", g_out == e_out, flush=True)
print("QWEN30B_BENCH_DONE")
