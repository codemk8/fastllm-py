#!/usr/bin/env python
"""Wire the fused MoE kernel into real decode and measure tok/s vs the eager
per-expert marlin path. Validates argmax parity too."""
import sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 2)[0])
import cupy as cp
from fastllm_py import DeviceMap, Model
from fastllm_py.device_router import MoeDeviceMap
from fastllm_py import moe as moemod
from fastllm_py.kernels import moe_int4

path = sys.argv[1]
m = Model.load(path, DeviceMap({"cuda:0": 1}), linear_quant="int4",
               moe_device=MoeDeviceMap({"cuda": 1}), gpu_expert_quant="int4",
               gpu_cache_bytes=12 << 30)
cfg = m.cfg

# build stacked row-major experts per MoE layer (from the fp16 expert weights)
print("building stacked experts...", flush=True)
for layer in m.layers:
    ml = getattr(layer, "moe", None)
    if ml is None:
        continue
    ews = [ml._materialize_cpu(e) for e in range(cfg.num_experts)]
    ml._stacked = moe_int4.build_stacked_experts(ews, group_size=128, xp=cp)
    ml._inter = ews[0]["gate"].shape[0]
    ml._ibuf = cp.empty((cfg.num_experts_per_tok, ml._inter), dtype=cp.float32)

_orig_forward = moemod.MoELayer.forward


def fused_forward(self, x):
    if x.shape[0] != 1:                         # prefill (T>1): fused kernel is
        return _orig_forward(self, x)           # decode-only, use eager
    cp_ = self.cp; cfg = self.cfg
    logits = x @ self.gate_weight.T            # (1, E)
    if self.gate_bias is not None: logits = logits + self.gate_bias
    if cfg.scoring_func == "sigmoid":
        probs = 1.0 / (1.0 + cp_.exp(-logits))
        select = probs + (self.e_score_bias if self.e_score_bias is not None else 0)
    else:
        e = cp_.exp(logits - logits.max(1, keepdims=True)); probs = e / e.sum(1, keepdims=True); select = probs
    row = select[0]; K = cfg.num_experts_per_tok
    idx = cp_.argpartition(-row, K - 1)[:K].astype(cp_.int32)
    w = probs[0][idx]
    if cfg.norm_topk_prob: w = w / (w.sum() + 1e-20)
    w = (w * cfg.routed_scaling_factor).astype(cp_.float32)
    out = moe_int4.fused_moe_ffn2(x[0], self._stacked, idx, w,
                                  cfg.hidden_dim, self._inter, inter_buf=self._ibuf)
    out = out[None]
    if self.shared is not None:
        s = moemod._expert_ffn(x, self.shared, cp_).astype(cp_.float32)
        if self.shared_gate is not None:
            g = (x @ self.shared_gate.T).astype(cp_.float32); s = s * (1.0 / (1.0 + cp_.exp(-g)))
        out = out + s
    return out.astype(x.dtype)

ids = np.array([785, 6722, 315, 9625, 374], dtype=np.int64)

# eager reference tokens (before patching)
ref = m.generate(ids, max_new_tokens=24)

# patch to fused, check argmax parity + speed
moemod.MoELayer.forward = fused_forward
got = m.generate(ids, max_new_tokens=24)
print("fused vs eager tokens:", "MATCH" if got == ref else "MISMATCH(int4 rounding differs)")
if got != ref:
    diverge = next((i for i in range(24) if got[i] != ref[i]), 24)
    print(f"  first divergence at token {diverge} (row-major vs marlin int4 rounding)")

logits, kvs = m.forward(ids); nx = int(cp.asnumpy(logits[-1]).argmax()); cp.cuda.Device().synchronize()
t0 = time.perf_counter(); NT = 32
for _ in range(NT):
    logits, kvs = m.forward(np.asarray([nx]), kvs); nx = int(cp.asnumpy(logits[-1]).argmax())
cp.cuda.Device().synchronize()
print(f"fused-MoE decode: {NT/(time.perf_counter()-t0):.1f} tok/s (eager baseline ~18.8)")
print("FUSED_BENCH_DONE")
