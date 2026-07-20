"""Gate computation, top-k expert selection, and CPU/GPU task splitting.

Mirrors fastllm's MergeMOE flow: after top-k routing, each selected expert
has a task list of (token_idx, weight). The splitter sends experts to GPU
or CPU based on placement + a benchmarked threshold (tokens-per-expert
below the threshold run faster on CPU; above it, GPU wins on throughput).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ExpertTask:
    expert_id: int
    token_idx: np.ndarray  # (n,) int32 — rows of x this expert processes
    weights: np.ndarray    # (n,) float32 — routing weights


def route_topk(scores: np.ndarray, top_k: int, *, norm_topk_prob: bool = False,
               scoring: str = "softmax", routed_scaling: float = 1.0,
               e_score_bias: np.ndarray | None = None) -> list[ExpertTask]:
    """scores: (T, num_experts) raw gate logits (fp32, on CPU).

    Returns one ExpertTask per activated expert. Handles both softmax
    (Qwen/Mixtral) and sigmoid+bias (DeepSeek V3) scoring.
    """
    T, E = scores.shape
    if scoring == "sigmoid":
        probs = 1.0 / (1.0 + np.exp(-scores))
        select = probs + (e_score_bias if e_score_bias is not None else 0.0)
    else:
        m = scores.max(-1, keepdims=True)
        e = np.exp(scores - m)
        probs = e / e.sum(-1, keepdims=True)
        select = probs

    topk_idx = np.argpartition(-select, top_k - 1, axis=-1)[:, :top_k]  # (T, k)
    topk_w = np.take_along_axis(probs, topk_idx, axis=-1)
    if norm_topk_prob:
        topk_w = topk_w / (topk_w.sum(-1, keepdims=True) + 1e-20)
    topk_w = topk_w * routed_scaling

    tasks = []
    flat_e = topk_idx.ravel()
    flat_t = np.repeat(np.arange(T, dtype=np.int32), top_k)
    flat_w = topk_w.ravel().astype(np.float32)
    order = np.argsort(flat_e, kind="stable")
    fe, ft, fw = flat_e[order], flat_t[order], flat_w[order]
    bounds = np.searchsorted(fe, np.arange(E + 1))
    for eid in np.unique(fe):
        lo, hi = bounds[eid], bounds[eid + 1]
        tasks.append(ExpertTask(int(eid), ft[lo:hi].copy(), fw[lo:hi].copy()))
    return tasks


@dataclass
class SpeedEstimator:
    """Decides the tokens-per-expert threshold below which CPU wins.

    Port of fastllm's benchmark: measure per-expert GEMV time on both
    devices for a few batch sizes, pick the crossover.
    """

    cpu_us_per_token: float = 0.0
    gpu_fixed_us: float = 0.0  # launch + upload overhead per expert
    gpu_us_per_token: float = 0.0
    threshold: int = 4  # default before calibration

    def calibrate(self, run_cpu, run_gpu, batch_sizes=(1, 2, 4, 8, 16, 32)):
        import time

        cpu_t, gpu_t = [], []
        for b in batch_sizes:
            t0 = time.perf_counter(); run_cpu(b); cpu_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); run_gpu(b); gpu_t.append(time.perf_counter() - t0)
        # crossover: first batch size where GPU beats CPU
        self.threshold = next(
            (b for b, c, g in zip(batch_sizes, cpu_t, gpu_t) if g < c),
            batch_sizes[-1],
        )
        return self.threshold


@dataclass
class ExpertPlacement:
    """Where each expert's weights live, per layer: 'cuda:0' | 'cpu' | 'disk'."""

    placement: dict[int, str] = field(default_factory=dict)  # expert_id -> device

    def split(self, tasks: list[ExpertTask], estimator: SpeedEstimator,
              gpu_cache_contains=None) -> tuple[list[ExpertTask], list[ExpertTask]]:
        """Returns (cpu_tasks, gpu_tasks).

        GPU-resident experts always run on GPU. CPU/disk-resident experts
        run on GPU only when their token count clears the upload-cost
        threshold OR they are already in the GPU cache.
        """
        cpu_set, gpu_set = [], []
        for t in tasks:
            dev = self.placement.get(t.expert_id, "cpu")
            if dev.startswith("cuda"):
                gpu_set.append(t)
            elif gpu_cache_contains and gpu_cache_contains(t.expert_id):
                gpu_set.append(t)
            elif len(t.token_idx) >= estimator.threshold:
                gpu_set.append(t)
            else:
                cpu_set.append(t)
        return cpu_set, gpu_set
