# CUDA-graph decode: speedup vs eager

INT4 (Marlin) dense models, single RTX 4090, single-stream greedy decode
(short prompt, 64 decode steps). Graph decode captures the whole per-token
step and replays it as one launch; output is **bit-exact** vs eager (verified
per-step, and gated at runtime by `verify()` with eager fallback).

| Model | Arch | eager tok/s | graph tok/s | speedup |
|---|---|---|---|---|
| Qwen3-0.6B | 28L h1024 | 43 | **213** | 4.9× |
| deepseek-coder-1.3b | 24L h2048 | 60 | **277** | 4.6× |
| R1-Distill-Qwen-1.5B | 28L h1536 | 49 | **205** | 4.2× |
| Qwen3-8B | 36L h4096 | 34 | **89** | 2.6× |

The speedup shrinks with model size: decode is dispatch-bound, and larger
GEMMs (bigger hidden dim) spend proportionally more time in the kernels and
less in Python/driver launch overhead — so collapsing the launches helps less.
The absolute numbers are strong (89 tok/s for 8B INT4 on one 4090).

Scope: dense non-MLA, single GPU (Marlin-quantized linears). MLA and MoE decode
still need the routing/latent-cache work before they're capturable; the 67B is
2-GPU so it needs multi-graph capture. See `docs/next-optimizations.md`.

Reproduce: `python scripts/test_graph_decode.py <int4-capable model dir>`.
