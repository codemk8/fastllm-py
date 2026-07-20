# Benchmark results

_Last updated: 2026-07-20 06:43 UTC; single-stream greedy decode, 128-token prefill, 64 decode steps; dev box: RTX 4090 (24 GB), NFS storage._

| Model | Variant | Arch | Load s | Prefill tok/s | Decode tok/s (p50) | GPU GB | Cache hit |
|---|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | fp16-experts | 27L h2048 E64k6 MLA | 416.3 | 21.0 | 0.8 (0.79) | 20.9 | 0.789 |
| Qwen1.5-MoE-A2.7B | fp16-experts | 24L h2048 E60k4 | 382.1 | 20.4 | 1.1 (1.15) | 23.0 | 0.781 |
| Qwen1.5-MoE-A2.7B | int4-experts | 24L h2048 E60k4 | 157.5 | 18.4 | 4.95 (5.66) | 18.8 | 0.871 |
| Qwen3-0.6B | default | 28L h1024 dense | 1.2 | 584.3 | 43.05 (43.58) | 6.2 | — |
| R1-Distill-Qwen-1.5B | default | 28L h1536 dense | 37.8 | 623.8 | 46.01 (46.62) | 10.8 | — |
| deepseek-coder-1.3b | default | 24L h2048 dense | 33.8 | 79.3 | 59.71 (60.92) | 9.1 | — |
| deepseek-llm-67b | int4-2gpu | 95L h8192 dense (GQA 64/8) | 3232 | 3.2 | 12.93 (13.23) | 39.6 | None |
| deepseek-llm-7b | fp16 | 30L h4096 dense | 23.7 | 75.0 | 36.55 (37.26) | 19.5 | — |
| deepseek-llm-7b | fp32-2gpu | 30L h4096 dense | 19.3 | 458.1 | 29.55 (29.8) | 21.3 | — |
| deepseek-moe-16b | fp16-experts | 28L h2048 E64k6 | 411.5 | 16.8 | 0.77 (0.78) | 22.4 | 0.776 |
| deepseek-moe-16b | int4-experts | 28L h2048 E64k6 | 181.3 | 18.7 | 3.17 (3.55) | 19.8 | 0.895 |
