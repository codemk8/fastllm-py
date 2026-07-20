# Benchmark results

_Last updated: 2026-07-20 07:32 UTC; single-stream greedy decode, 128-token prefill, 64 decode steps; dev box: RTX 4090 (24 GB), NFS storage._

| Model | Variant | Arch | Load s | Prefill tok/s | Decode tok/s (p50) | GPU GB | Cache hit |
|---|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | fp16-experts | 27L h2048 E64k6 MLA | 44.6 | 19.9 | 0.82 (0.82) | 18.4 | 0.789 |
| DeepSeek-V2-Lite | int4-experts | 27L h2048 E64k6 MLA | 158.3 | 20.7 | 4.23 (4.61) | 15.5 | 0.897 |
| Qwen1.5-MoE-A2.7B | fp16-experts | 24L h2048 E60k4 | 393.3 | 19.5 | 1.16 (1.16) | 20.3 | 0.781 |
| Qwen1.5-MoE-A2.7B | int4-experts | 24L h2048 E60k4 | 159.0 | 17.7 | 6.39 (7.49) | 16.1 | 0.871 |
| Qwen3-0.6B | default | 28L h1024 dense | 1.2 | 584.3 | 43.05 (43.58) | 6.2 | — |
| R1-Distill-Qwen-1.5B | default | 28L h1536 dense | 37.8 | 623.8 | 46.01 (46.62) | 10.8 | — |
| deepseek-coder-1.3b | default | 24L h2048 dense | 33.8 | 79.3 | 59.71 (60.92) | 9.1 | — |
| deepseek-llm-67b | int4-2gpu | 95L h8192 dense (GQA 64/8) | 3232 | 3.2 | 12.93 (13.23) | 39.6 | None |
| deepseek-llm-7b | fp16 | 30L h4096 dense | 23.7 | 75.0 | 36.55 (37.26) | 19.5 | — |
| deepseek-llm-7b | fp32-2gpu | 30L h4096 dense | 19.3 | 458.1 | 29.55 (29.8) | 21.3 | — |
| deepseek-moe-16b | fp16-experts | 28L h2048 E64k6 | 411.5 | 16.8 | 0.77 (0.78) | 22.4 | 0.776 |
| deepseek-moe-16b | int4-experts | 28L h2048 E64k6 | 836.5 | 19.9 | 4.07 (4.49) | 16.2 | 0.895 |
