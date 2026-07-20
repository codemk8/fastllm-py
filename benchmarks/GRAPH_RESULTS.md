# CUDA-graph decode: speedup vs eager

INT4 (Marlin) dense models, RTX 4090(s), single-stream greedy decode (short
prompt, 64 decode steps). Graph decode captures the whole per-token step and
replays it as one launch; output is **bit-exact** vs eager (verified per-step
and gated at runtime by `verify()` with eager fallback). Multi-GPU models are
captured as one graph per device-segment, chained by an async boundary-hidden
copy.

| Model | Arch | GPUs | eager tok/s | graph tok/s | speedup |
|---|---|---|---|---|---|
| Qwen3-0.6B | 28L h1024 | 1 | 44 | **228** | 5.2× |
| deepseek-coder-1.3b | 24L h2048 | 1 | 60 | **277** | 4.6× |
| R1-Distill-Qwen-1.5B | 28L h1536 | 1 | 49 | **205** | 4.2× |
| Qwen3-8B | 36L h4096 | 1 | 33 | **89** | 2.7× |
| deepseek-llm-67b | 95L h8192 | 2 | 15.4 | **20.7** | 1.34× |

The speedup shrinks with model size: decode is dispatch-bound, and larger GEMMs
(bigger hidden dim) spend proportionally more time in the kernels and less in
Python/driver launch overhead — so collapsing the launches helps less. Still a
win everywhere, and the absolute numbers are strong (89 tok/s for 8B INT4 on
one 4090; 19 tok/s for 67B INT4 on two).

**Attention is a flash-decode RawKernel, O(valid_len).** Attention loops only
over the valid `[0, pos]` keys (`valid_len = pos+1`, read from the device pos
buffer), one block per head with online softmax — so its cost is independent of
the KV buffer size. An earlier version used a cupy reduction over the *whole*
`max_len` buffer (O(max_len)); at `max_len=2048` that made the 8192-wide 67B
graph 0.51× (slower than eager). The kernel removed that entirely: the 67B is
1.34× at `max_len=4096` just as at `max_len=128`. `GraphDecoder` still buckets
`max_len` to `prompt+max_new` (and re-captures on growth) purely to save
memory — it no longer affects speed.

Scope: dense non-MLA, any GPU count (Marlin-quantized linears). MLA decode and
*offloaded* MoE still fall back to eager. For MoE that fits in VRAM at INT4, the
win is expert **residency** (`moe_device={"cuda":1}` + `gpu_expert_quant="int4"`),
not graph capture — see `docs/next-optimizations.md`.

Reproduce: `python scripts/test_graph_decode.py <int4 model>` (1 GPU),
`python scripts/run_67b_graph.py <67b dir>` (2 GPU).
