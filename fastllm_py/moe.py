"""Hybrid CPU+GPU MoE forward — the core of fastllm's strategy.

Per token batch:
  1. gate on GPU -> logits pulled to CPU (tiny)
  2. route_topk -> per-expert task lists
  3. split into cpu_set / gpu_set (placement + speed threshold + cache)
  4. GPU experts run on a non-blocking stream; CPU experts run in a
     thread pool concurrently (BLAS releases the GIL)
  5. CPU partial result is DMA'd to GPU (pinned staging) and added
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .expert_cache import GpuExpertCache
from .expert_router import ExpertPlacement, ExpertTask, SpeedEstimator, route_topk
from .kernels.ops import swiglu


def _expert_ffn(x, w, xp):
    """x: (n, hidden); w: {gate, up, down} weight matrices (out, in)."""
    g = x @ w["gate"].T
    u = x @ w["up"].T
    return swiglu(g, u) @ w["down"].T


# ---------------------------------------------------------------- marlin int4
def quantize_marlin_matrix(w, group_size: int = 128) -> dict:
    """Quantize one (out, in) float matrix to upload-ready Marlin INT4:
    {"qweight": uint32, "scales": fp16, "zeros": uint32} numpy arrays.
    Repack and permutations are done here once (device repack, downloaded),
    so runtime upload is a plain memcpy at ~1/4 the fp16 size."""
    import cupy as cp

    from .kernels import marlin

    size_n, size_k = w.shape
    g = w.astype(np.float32).reshape(size_n, size_k // group_size, group_size)
    wmin, wmax = g.min(axis=2), g.max(axis=2)
    scale = (wmax - wmin) / 15.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    zero = np.clip(np.rint(-wmin / scale), 0, 15).astype(np.int64)
    q = np.clip(np.rint(g / scale[:, :, None]) + zero[:, :, None], 0, 15
                ).astype(np.uint8).reshape(size_n, size_k)
    packed = marlin.pack_gptq_qweight(q)
    repacked = cp.asnumpy(marlin.marlin_repack(cp.asarray(packed), size_k, size_n))
    scales, zeros = marlin.build_marlin_scales_zeros(scale, zero, group_size)
    return {"qweight": repacked, "scales": scales, "zeros": zeros}


def build_marlin_expert_payload(w_fp16: dict, group_size: int = 128) -> dict:
    """Quantize one expert's {gate,up,down}: flat "<proj>.qweight/..." dict."""
    payload = {}
    for proj, w in w_fp16.items():
        for k, v in quantize_marlin_matrix(w, group_size).items():
            payload[f"{proj}.{k}"] = v
    return payload


def _expert_ffn_marlin(x_fp16, w):
    """Marlin INT4 expert ffn. w: uploaded payload dict (GPU arrays).

    Uses marlin.gemm_fast (lean, no per-call conversions/validation) since
    payloads are pre-validated + contiguous. Ordering is provided by the
    caller's per-layer compute_stream.synchronize(), not per-GEMM syncs.
    """
    from .kernels.marlin import gemm_fast

    hidden = x_fp16.shape[1]
    inter = w["gate.scales"].shape[1]
    g = gemm_fast(x_fp16, w["gate.qweight"], w["gate.scales"], w["gate.zeros"],
                  inter, hidden)
    u = gemm_fast(x_fp16, w["up.qweight"], w["up.scales"], w["up.zeros"],
                  inter, hidden)
    act = swiglu(g, u)
    return gemm_fast(act, w["down.qweight"], w["down.scales"], w["down.zeros"],
                     hidden, inter)


class MoELayer:
    def __init__(self, cfg, layer_idx: int, gate_weight, expert_weights_cpu,
                 placement: ExpertPlacement, cache: GpuExpertCache,
                 estimator: SpeedEstimator, shared_weights=None, shared_gate=None,
                 gate_bias=None, e_score_bias=None, pool: ThreadPoolExecutor | None = None,
                 gpu_payloads: dict | None = None):
        import cupy as cp

        self.cp = cp
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.gate_weight = gate_weight          # (E, hidden) on GPU
        self.gate_bias = gate_bias
        self.e_score_bias = e_score_bias        # DeepSeek V3 e_score_correction_bias
        self.experts_cpu = expert_weights_cpu   # eid -> {gate,up,down} np arrays or callable
        self.gpu_payloads = gpu_payloads        # eid -> marlin int4 payload (or None=fp16)
        self.placement = placement
        self.cache = cache
        self.estimator = estimator
        self.shared = shared_weights            # dense shared expert(s) on GPU
        self.shared_gate = shared_gate          # qwen2_moe: sigmoid gate (1, hidden)
        import os

        self.pool = pool or ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))
        # exponentially-decayed expert activation frequency (for prefetch)
        self.freq = np.zeros(cfg.num_experts, dtype=np.float64)
        # blocking stream: implicit ordering vs null-stream work (attention),
        # still overlaps with the cache's copy stream and CPU threads
        self.compute_stream = cp.cuda.Stream()

    # ---- device-side helpers -------------------------------------------
    def _payload_source(self, eid: int):
        """What gets uploaded for expert eid: marlin payload or fp16 dict."""
        if self.gpu_payloads is not None:
            return lambda: self.gpu_payloads[eid]
        return lambda: self._materialize_cpu(eid)

    def _gpu_expert_weights(self, eid: int):
        return self.cache.get_or_upload((self.layer_idx, eid),
                                        self._payload_source(eid))

    def _materialize_cpu(self, eid: int):
        w = self.experts_cpu[eid]
        return w() if callable(w) else w

    def _run_gpu_experts(self, x_gpu, tasks: list[ExpertTask], out_gpu):
        cp = self.cp
        marlin_mode = self.gpu_payloads is not None
        with self.compute_stream:
            x16 = x_gpu.astype(cp.float16) if marlin_mode else None
            for t in tasks:
                w = self._gpu_expert_weights(t.expert_id)
                single = t.token_idx.shape[0] == 1
                idx = int(t.token_idx[0]) if single else cp.asarray(t.token_idx)
                xin = (x16 if marlin_mode else x_gpu)[idx]
                if single:
                    xin = xin[None]  # (1, hidden)
                wgt = cp.asarray(t.weights)[:, None]
                if marlin_mode:
                    y = _expert_ffn_marlin(xin, w).astype(cp.float32) * wgt
                else:
                    y = _expert_ffn(xin, w, cp) * wgt
                y = y.astype(out_gpu.dtype)
                if single:  # decode: single token, direct indexed add
                    out_gpu[idx] += y[0]
                else:  # prefill: unique indices within an expert
                    cp.add.at(out_gpu, idx, y)

    def _run_cpu_experts(self, x_cpu, tasks: list[ExpertTask]):
        out = np.zeros_like(x_cpu, dtype=np.float32)

        def one(t: ExpertTask):
            w = self._materialize_cpu(t.expert_id)
            if next(iter(w.values())).dtype == np.float16:
                # numpy skips BLAS on mixed fp32@fp16 (falls back to a scalar
                # loop, ~10x slower than the cast+sgemm path)
                w = {k: v.astype(np.float32) for k, v in w.items()}
            y = _expert_ffn(x_cpu[t.token_idx], w, np) * t.weights[:, None]
            return t.token_idx, y

        for idx, y in self.pool.map(one, tasks):
            np.add.at(out, idx, y.astype(np.float32))
        return out

    def prefetch_predicted(self, top_n: int = 8):
        """Prefetch this layer's historically hottest experts into the GPU
        cache on the copy stream. Call while an earlier layer computes."""
        if not self.freq.any():
            return
        if self.cache.used > 0.9 * self.cache.max_bytes:
            return  # near-full: prefetch would evict (and eviction syncs)
        for eid in np.argsort(-self.freq)[:top_n]:
            eid = int(eid)
            key = (self.layer_idx, eid)
            if key not in self.cache:
                self.cache.prefetch(key, self._payload_source(eid))

    # ---- forward --------------------------------------------------------
    def forward(self, x):
        """x: (T, hidden) on GPU. Returns (T, hidden) on GPU."""
        cp = self.cp
        cfg = self.cfg
        T = x.shape[0]

        logits = x @ self.gate_weight.T
        if self.gate_bias is not None:
            logits = logits + self.gate_bias
        scores_cpu = cp.asnumpy(logits.astype(cp.float32))

        tasks = route_topk(
            scores_cpu, cfg.num_experts_per_tok,
            norm_topk_prob=cfg.norm_topk_prob, scoring=cfg.scoring_func,
            routed_scaling=cfg.routed_scaling_factor,
            e_score_bias=self.e_score_bias,
        )
        self.freq *= 0.98
        for t in tasks:
            self.freq[t.expert_id] += len(t.token_idx)

        cpu_set, gpu_set = self.placement.split(
            tasks, self.estimator,
            gpu_cache_contains=lambda eid: (self.layer_idx, eid) in self.cache,
        )

        out_gpu = cp.zeros((T, x.shape[1]), dtype=cp.float32)

        # launch GPU work first (async on its stream) ...
        cpu_future = None
        if cpu_set:
            x_cpu = cp.asnumpy(x.astype(cp.float32))
            cpu_future = self.pool.submit(self._run_cpu_experts, x_cpu, cpu_set)
        if gpu_set:
            self._run_gpu_experts(x, gpu_set, out_gpu)
        if self.shared is not None:
            with self.compute_stream:
                s = _expert_ffn(x, self.shared, cp).astype(cp.float32)
                if self.shared_gate is not None:
                    g = (x @ self.shared_gate.T).astype(cp.float32)
                    s = s * (1.0 / (1.0 + cp.exp(-g)))
                out_gpu += s

        if cpu_future is not None:
            cpu_out = cpu_future.result()
            with self.compute_stream:
                out_gpu += cp.asarray(cpu_out)  # DMA + add, ordered on stream
        self.compute_stream.synchronize()
        return out_gpu.astype(x.dtype)
