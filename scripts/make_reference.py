#!/usr/bin/env python
"""Generate HF reference logits for a model: saves token ids + fp32 logits.

Usage: make_reference.py <model_path> <out.npz> [prompt]
Runs on CPU in float32 for maximum numeric fidelity (needs RAM ~= 4 bytes/param).
"""
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TRUST = os.environ.get("TRUST_REMOTE_CODE", "0") == "1"


def main():
    model_path, out_path = sys.argv[1], sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) > 3 else "The capital of France is"

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=TRUST)
    ids = tok(prompt).input_ids

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        trust_remote_code=TRUST)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(ids)[None]).logits[0].float().numpy()
        # also record 8 greedy continuation tokens for a generation check
        out = model.generate(torch.tensor(ids)[None], max_new_tokens=8,
                             do_sample=False, temperature=None, top_p=None, top_k=None)
    np.savez(out_path, ids=np.asarray(ids, dtype=np.int64), logits=logits,
             greedy=out[0, len(ids):].numpy())
    print("saved", out_path, "logits", logits.shape)


if __name__ == "__main__":
    main()
