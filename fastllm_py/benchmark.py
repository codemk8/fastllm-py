"""Runtime calibration of the CPU/GPU expert split threshold.

Port of fastllm's MoeExpertSpeedEstimator: micro-benchmark one representative
expert at geometric batch sizes on CPU (numpy in the worker pool) and GPU
(upload + ffn, since non-cached experts pay the DMA), then pick the smallest
token count where GPU wins.
"""
from __future__ import annotations

import time

import numpy as np


def calibrate_model(model, batch_sizes=(1, 2, 4, 8, 16, 32, 64, 128),
                    repeats: int = 5, verbose: bool = True) -> int:
    """Calibrates and installs the threshold on the model's SpeedEstimator."""
    import cupy as cp

    from .moe import _expert_ffn

    moe_layer = next((l.moe for l in model.layers if l.moe is not None), None)
    if moe_layer is None:
        return 0
    w_cpu = moe_layer._materialize_cpu(0)
    hidden = w_cpu["gate"].shape[1]
    rng = np.random.default_rng(0)
    x_full = rng.standard_normal((max(batch_sizes), hidden)).astype(np.float32)
    x_gpu_full = cp.asarray(x_full)
    stream = cp.cuda.Stream()

    def gpu_once(b):
        with stream:
            w_gpu = {k: cp.asarray(v) for k, v in w_cpu.items()}  # upload cost
            _expert_ffn(x_gpu_full[:b], w_gpu, cp)
        stream.synchronize()

    def cpu_once(b):
        w = w_cpu
        if next(iter(w.values())).dtype == np.float16:
            w = {k: v.astype(np.float32) for k, v in w.items()}  # match runtime path
        _expert_ffn(x_full[:b], w, np)

    gpu_once(1)  # warmup / JIT
    cpu_once(1)

    threshold = batch_sizes[-1]
    rows = []
    for b in batch_sizes:
        t0 = time.perf_counter()
        for _ in range(repeats):
            cpu_once(b)
        t_cpu = (time.perf_counter() - t0) / repeats
        t0 = time.perf_counter()
        for _ in range(repeats):
            gpu_once(b)
        t_gpu = (time.perf_counter() - t0) / repeats
        rows.append((b, t_cpu * 1e3, t_gpu * 1e3))
        if t_gpu < t_cpu and threshold == batch_sizes[-1]:
            threshold = b
    if verbose:
        for b, tc, tg in rows:
            mark = "→GPU" if b >= threshold else "→CPU"
            print(f"  batch {b:4d}: cpu {tc:7.3f}ms  gpu(+upload) {tg:7.3f}ms  {mark}")
        print(f"  calibrated expert split threshold: {threshold} tokens")

    model._moe_shared["estimator"].threshold = threshold
    return threshold
