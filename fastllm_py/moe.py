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


class MoELayer:
    def __init__(self, cfg, layer_idx: int, gate_weight, expert_weights_cpu,
                 placement: ExpertPlacement, cache: GpuExpertCache,
                 estimator: SpeedEstimator, shared_weights=None,
                 gate_bias=None, e_score_bias=None, pool: ThreadPoolExecutor | None = None):
        import cupy as cp

        self.cp = cp
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.gate_weight = gate_weight          # (E, hidden) on GPU
        self.gate_bias = gate_bias
        self.e_score_bias = e_score_bias        # DeepSeek V3 e_score_correction_bias
        self.experts_cpu = expert_weights_cpu   # eid -> {gate,up,down} np arrays or callable
        self.placement = placement
        self.cache = cache
        self.estimator = estimator
        self.shared = shared_weights            # dense shared expert(s) on GPU
        import os

        self.pool = pool or ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))
        self.compute_stream = cp.cuda.Stream(non_blocking=True)

    # ---- device-side helpers -------------------------------------------
    def _gpu_expert_weights(self, eid: int):
        key = (self.layer_idx, eid)
        return self.cache.get_or_upload(key, lambda: self._materialize_cpu(eid))

    def _materialize_cpu(self, eid: int):
        w = self.experts_cpu[eid]
        return w() if callable(w) else w

    def _run_gpu_experts(self, x_gpu, tasks: list[ExpertTask], out_gpu):
        cp = self.cp
        with self.compute_stream:
            for t in tasks:
                w = self._gpu_expert_weights(t.expert_id)
                idx = cp.asarray(t.token_idx)
                wgt = cp.asarray(t.weights)[:, None]
                y = _expert_ffn(x_gpu[idx], w, cp) * wgt
                # scatter-add (indices within one expert are unique)
                cp.add.at(out_gpu, idx, y.astype(out_gpu.dtype))

    def _run_cpu_experts(self, x_cpu, tasks: list[ExpertTask]):
        out = np.zeros_like(x_cpu, dtype=np.float32)

        def one(t: ExpertTask):
            w = self._materialize_cpu(t.expert_id)
            y = _expert_ffn(x_cpu[t.token_idx], w, np) * t.weights[:, None]
            return t.token_idx, y

        for idx, y in self.pool.map(one, tasks):
            np.add.at(out, idx, y.astype(np.float32))
        return out

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
                out_gpu += _expert_ffn(x, self.shared, cp).astype(cp.float32)

        if cpu_future is not None:
            cpu_out = cpu_future.result()
            out_gpu += cp.asarray(cpu_out)  # async DMA then add
        self.compute_stream.synchronize()
        return out_gpu.astype(x.dtype)
