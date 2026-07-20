#!/usr/bin/env python
"""First INT4 load + smoke test of deepseek-llm-67b-chat across 2×4090.

First run quantizes all eligible projections to Marlin INT4 and fills
<model>/.marlin_cache (slow, one-time); later runs load from cache.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def main():
    import cupy as cp
    from transformers import AutoTokenizer

    from fastllm_py import DeviceMap, Model

    path = "models/deepseek-llm-67b-chat"
    prompt = "Q: What is the capital of France?\nA:"

    tok = AutoTokenizer.from_pretrained(path)
    ids = np.asarray(tok(prompt).input_ids, dtype=np.int64)

    t0 = time.time()
    model = Model.load(path, DeviceMap({"cuda:0": 1, "cuda:1": 1}),
                       linear_quant="int4")
    load_s = time.time() - t0
    cache_gb = sum(f.stat().st_size for f in
                   (Path(path) / ".marlin_cache").glob("*.npz")) / 2**30
    print(f"load: {load_s:.0f}s | marlin_cache {cache_gb:.1f}GB", flush=True)
    for d in (0, 1):
        free, total = cp.cuda.Device(d).mem_info
        print(f"  cuda:{d} used {(total-free)/2**30:.1f}GB", flush=True)

    # prefill
    t0 = time.time()
    logits, kvs = model.forward(ids)
    cp.cuda.Device(1).synchronize()
    print(f"prefill {len(ids)} tok: {time.time()-t0:.2f}s "
          f"({len(ids)/(time.time()-t0):.1f} tok/s)", flush=True)

    # decode
    nxt = int(cp.asnumpy(logits[-1]).argmax())
    out, times = [nxt], []
    for _ in range(24):
        t0 = time.time()
        logits, kvs = model.forward(np.asarray([nxt]), kvs)
        nxt = int(cp.asnumpy(logits[-1]).argmax())
        times.append(time.time() - t0)
        out.append(nxt)
    times = np.asarray(times)
    print(f"decode: {1/times.mean():.2f} tok/s (p50 {1/np.median(times):.2f})",
          flush=True)
    print("GEN:", repr(tok.decode(ids.tolist() + out)), flush=True)
    print("RUN_67B_DONE", flush=True)


if __name__ == "__main__":
    main()
