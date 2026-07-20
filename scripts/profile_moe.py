#!/usr/bin/env python
"""Profile resident-MoE decode: where does a token's time go? Scopes the fused
kernel. Faithful single-pass timed copy of MoELayer.forward — times the
gate+routing (incl the D2H sync) phase vs the GPU expert-dispatch phase."""
import sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 2)[0])
import cupy as cp
from fastllm_py import DeviceMap, Model
from fastllm_py.device_router import MoeDeviceMap
from fastllm_py import moe as moemod
from fastllm_py.expert_router import route_topk

path = sys.argv[1]
m = Model.load(path, DeviceMap({"cuda:0": 1}), linear_quant="int4",
               moe_device=MoeDeviceMap({"cuda": 1}), gpu_expert_quant="int4",
               gpu_cache_bytes=12 << 30)

acc = {"route": 0.0, "experts": 0.0, "n": 0}


def timed_forward(self, x):
    cp_ = self.cp; cfg = self.cfg; T = x.shape[0]
    cp_.cuda.Device().synchronize(); t0 = time.perf_counter()
    logits = x @ self.gate_weight.T
    if self.gate_bias is not None: logits = logits + self.gate_bias
    scores_cpu = cp_.asnumpy(logits.astype(cp_.float32))          # D2H SYNC
    tasks = route_topk(scores_cpu, cfg.num_experts_per_tok,
                       norm_topk_prob=cfg.norm_topk_prob, scoring=cfg.scoring_func,
                       routed_scaling=cfg.routed_scaling_factor, e_score_bias=self.e_score_bias,
                       n_group=cfg.n_group, topk_group=cfg.topk_group)
    self.freq *= 0.98
    for t in tasks: self.freq[t.expert_id] += len(t.token_idx)
    cpu_set, gpu_set = self.placement.split(
        tasks, self.estimator,
        gpu_cache_contains=lambda eid: (self.layer_idx, eid) in self.cache)
    cp_.cuda.Device().synchronize(); t1 = time.perf_counter()

    out_gpu = cp_.zeros((T, x.shape[1]), dtype=cp_.float32)
    if gpu_set:
        self._run_gpu_experts(x, gpu_set, out_gpu)
    if self.shared is not None:
        with self.compute_stream:
            s = moemod._expert_ffn(x, self.shared, cp_).astype(cp_.float32)
            if self.shared_gate is not None:
                g = (x @ self.shared_gate.T).astype(cp_.float32)
                s = s * (1.0 / (1.0 + cp_.exp(-g)))
            out_gpu += s
    self.compute_stream.synchronize()
    cp_.cuda.Device().synchronize(); t2 = time.perf_counter()
    acc["route"] += t1 - t0; acc["experts"] += t2 - t1; acc["n"] += 1
    return out_gpu.astype(x.dtype)


moemod.MoELayer.forward = timed_forward

ids = np.array([785, 6722, 315, 9625, 374], dtype=np.int64)
logits, kvs = m.forward(ids); nx = int(cp.asnumpy(logits[-1]).argmax())
cp.cuda.Device().synchronize()
acc.update(route=0.0, experts=0.0, n=0)
NT = 32
t0 = time.perf_counter()
for _ in range(NT):
    logits, kvs = m.forward(np.asarray([nx]), kvs); nx = int(cp.asnumpy(logits[-1]).argmax())
cp.cuda.Device().synchronize()
total = time.perf_counter() - t0

print(f"decode: {total/NT*1000:.2f} ms/token  ({NT/total:.1f} tok/s)")
print(f"MoE layers/token: {acc['n']/NT:.0f}, experts/tok/layer: {m.cfg.num_experts_per_tok}"
      f" of {m.cfg.num_experts}")
print(f"  gate + routing + D2H sync : {acc['route']/NT*1000:7.2f} ms/tok ({acc['route']/total*100:4.0f}%)")
print(f"  GPU expert dispatch+GEMV  : {acc['experts']/NT*1000:7.2f} ms/tok ({acc['experts']/total*100:4.0f}%)")
print(f"  non-MoE (attn/norm/lmhead): {(total-acc['route']-acc['experts'])/NT*1000:7.2f} ms/tok "
      f"({(total-acc['route']-acc['experts'])/total*100:4.0f}%)")
print("PROFILE_DONE")
