#!/usr/bin/env python
"""Decode throughput benchmark.

Usage: benchmark_throughput.py <model_path> [--tokens 64] [--device ...]
       [--moe-device '{"cuda":1,"cpu":3}'] [--expert-dtype float16]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from transformers import AutoTokenizer

from fastllm_py import DeviceMap, Model
from fastllm_py.device_router import MoeDeviceMap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--prompt", default="Write a short story about a robot:")
    p.add_argument("--device", default='{"cuda:0": 1}')
    p.add_argument("--moe-device", default='{"cpu": 1}')
    p.add_argument("--expert-dtype", default="float16")
    p.add_argument("--gpu-cache-gb", type=float, default=8.0)
    p.add_argument("--calibrate", action="store_true",
                   help="calibrate CPU/GPU expert split threshold first")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = np.asarray(tok(args.prompt).input_ids, dtype=np.int64)

    t0 = time.time()
    model = Model.load(args.model, DeviceMap(json.loads(args.device)),
                       moe_device=MoeDeviceMap(json.loads(args.moe_device)),
                       expert_dtype=args.expert_dtype,
                       gpu_cache_bytes=int(args.gpu_cache_gb * 2**30))
    print(f"load: {time.time()-t0:.1f}s")

    if args.calibrate:
        from fastllm_py.benchmark import calibrate_model

        calibrate_model(model)

    # prefill
    t0 = time.time()
    logits, kvs = model.forward(ids)
    import cupy as cp

    cp.cuda.Device().synchronize()
    print(f"prefill {len(ids)} tokens: {time.time()-t0:.2f}s")

    # decode
    nxt = int(cp.asnumpy(logits[-1]).argmax())
    times = []
    out = [nxt]
    for i in range(args.tokens):
        t0 = time.time()
        logits, kvs = model.forward(np.asarray([nxt]), kvs)
        nxt = int(cp.asnumpy(logits[-1]).argmax())
        times.append(time.time() - t0)
        out.append(nxt)

    times = np.asarray(times)
    print(f"decode: {1/times.mean():.2f} tok/s  "
          f"(p50 {1/np.median(times):.2f}, worst {1/times.max():.2f})")
    if hasattr(model, "_moe_shared"):
        c = model._moe_shared["cache"]
        print(f"expert cache: hit_rate={c.hit_rate:.1%} "
              f"used={c.used/2**30:.2f}GiB misses={c.misses}")
    print(tok.decode(out))


if __name__ == "__main__":
    main()
