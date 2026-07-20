#!/usr/bin/env python
"""DeepSeek-V2 reference with OFFICIAL softmax-scale semantics.

transformers>=5 native DeepseekV2 omits the yarn mscale^2 softmax correction
that the official modeling_deepseek.py, vLLM, and fastllm all apply
(softmax_scale *= (0.1*mscale_all_dim*ln(factor)+1)^2). This script patches
the loaded HF model to official semantics before dumping reference logits.
"""
import math
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_path, out_path = sys.argv[1], sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) > 3 else "The capital of France is"

    tok = AutoTokenizer.from_pretrained(model_path)
    ids = tok(prompt).input_ids

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()

    rs = model.config.rope_parameters if hasattr(model.config, "rope_parameters") \
        else model.config.rope_scaling
    if rs and rs.get("mscale_all_dim"):
        m = 0.1 * float(rs["mscale_all_dim"]) * math.log(float(rs["factor"])) + 1.0
        for layer in model.model.layers:
            layer.self_attn.scaling *= m * m
        print(f"patched softmax scaling by mscale^2 = {m*m:.4f}")

    with torch.no_grad():
        logits = model(torch.tensor(ids)[None]).logits[0].float().numpy()
        out = model.generate(torch.tensor(ids)[None], max_new_tokens=8,
                             do_sample=False, temperature=None, top_p=None, top_k=None)
    np.savez(out_path, ids=np.asarray(ids, dtype=np.int64), logits=logits,
             greedy=out[0, len(ids):].numpy())
    print("saved", out_path, tok.decode(out[0]))


if __name__ == "__main__":
    main()
