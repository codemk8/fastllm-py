#!/usr/bin/env python
"""Quick generation demo: python scripts/generate.py models/Qwen3-0.6B "prompt" """
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from transformers import AutoTokenizer

from fastllm_py import DeviceMap, Model


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
    n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 32

    tok = AutoTokenizer.from_pretrained(model_path)
    ids = np.asarray(tok(prompt).input_ids, dtype=np.int64)

    t0 = time.time()
    model = Model.load(model_path, DeviceMap({"cuda:0": 1}), dtype="float32")
    print(f"[load {time.time()-t0:.1f}s] {model.cfg.model_type}: "
          f"{model.cfg.num_layers}L hidden={model.cfg.hidden_dim} "
          f"heads={model.cfg.num_heads}/{model.cfg.num_kv_heads} moe={model.cfg.is_moe}")

    t0 = time.time()
    out = model.generate(ids, max_new_tokens=n_new)
    dt = time.time() - t0
    print(tok.decode(ids.tolist() + out))
    print(f"\n[{n_new} tokens in {dt:.2f}s = {n_new/dt:.1f} tok/s]")


if __name__ == "__main__":
    main()
