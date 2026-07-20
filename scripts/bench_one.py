#!/usr/bin/env python
"""Benchmark a single model config; prints one JSON line to stdout.

Usage: bench_one.py '<json config>'
Config: {"name", "path", "dtype", "device", "moe_device", "expert_dtype",
         "gpu_expert_quant", "gpu_cache_gb", "prefill_tokens", "decode_tokens"}
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def main():
    cfg = json.loads(sys.argv[1])
    import cupy as cp
    from transformers import AutoTokenizer

    from fastllm_py import DeviceMap, Model
    from fastllm_py.device_router import MoeDeviceMap

    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
    # deterministic synthetic prompt of the requested length
    base = tok("The quick brown fox jumps over the lazy dog. ").input_ids
    n_prefill = cfg.get("prefill_tokens", 128)
    ids = np.asarray((base * (n_prefill // len(base) + 1))[:n_prefill], dtype=np.int64)

    t0 = time.time()
    model = Model.load(
        cfg["path"],
        DeviceMap(cfg.get("device", {"cuda:0": 1})),
        dtype=cfg.get("dtype", "float32"),
        moe_device=MoeDeviceMap(cfg["moe_device"]) if cfg.get("moe_device") else None,
        expert_dtype=cfg.get("expert_dtype", "float16"),
        gpu_cache_bytes=int(cfg.get("gpu_cache_gb", 8) * 2**30),
        gpu_expert_quant=cfg.get("gpu_expert_quant", "none"),
    )
    load_s = time.time() - t0

    # prefill
    t0 = time.time()
    logits, kvs = model.forward(ids)
    cp.cuda.Device().synchronize()
    prefill_s = time.time() - t0

    # drain pool high-water marks left by prefill: per-stream arenas
    # (default/compute/copy) each retain their peak, which can pin the
    # pool near the VRAM ceiling and make decode allocations thrash
    logits_host = cp.asnumpy(logits)
    del logits
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    logits = cp.asarray(logits_host)

    # decode
    nxt = int(cp.asnumpy(logits[-1]).argmax())
    out = [nxt]
    times = []
    n_decode = cfg.get("decode_tokens", 64)
    for _ in range(n_decode):
        t0 = time.time()
        logits, kvs = model.forward(np.asarray([nxt]), kvs)
        nxt = int(cp.asnumpy(logits[-1]).argmax())
        times.append(time.time() - t0)
        out.append(nxt)
    times = np.asarray(times)

    dev_id = 0
    free, total = cp.cuda.Device(dev_id).mem_info
    result = {
        "name": cfg["name"],
        "variant": cfg.get("variant", "default"),
        "model_type": model.cfg.model_type,
        "params_note": f"{model.cfg.num_layers}L h{model.cfg.hidden_dim}"
                       + (f" E{model.cfg.num_experts}k{model.cfg.num_experts_per_tok}"
                          if model.cfg.is_moe else " dense")
                       + (" MLA" if model.cfg.is_mla else ""),
        "load_s": round(load_s, 1),
        "prefill_tok_s": round(n_prefill / prefill_s, 1),
        "decode_tok_s": round(1 / times.mean(), 2),
        "decode_p50": round(1 / float(np.median(times)), 2),
        "decode_worst": round(1 / times.max(), 2),
        "gpu_mem_gb": round((total - free) / 2**30, 1),
        "sample": tok.decode(out[:24]),
    }
    if hasattr(model, "_moe_shared"):
        c = model._moe_shared["cache"]
        result["cache_hit"] = round(c.hit_rate, 3)
    print("BENCH_RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
