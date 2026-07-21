# Handoff: ktransformers vs fastllm-py — fair INT4 decode comparison on the 4090 host

## Mission
Run a **fair, same-box INT4 decode-throughput comparison** between **ktransformers**
and **fastllm-py** (this repo) on the 4090 host, using **Qwen3-30B-A3B**. Report
decode tok/s for both, same model, same precision class, same GPU(s). A parallel
agent on a 5090 box already benchmarked our side and tried ktransformers there —
read "What's already known" so you don't repeat dead ends.

## ⚠️ FIRST — gate check (do this before anything else)
ktransformers' MoE **unconditionally constructs a CPUInfer backend whose native
code uses AVX512**. On the 5090 box the CPU was an Arrow Lake i9 (AVX2-only) and
ktransformers **hard-crashed with SIGILL** (`kt_kernel/experts_base.py:177
_get_cpu_infer`) — even with all experts forced onto the GPU. So:

```bash
grep -c avx512f /proc/cpuinfo
```
- **If 0 → STOP.** ktransformers cannot run on this host either; report back and
  we'll need a different box. Only run the fastllm side.
- **If ≥1 → proceed.** This host can run ktransformers (presumably why we're here).

Also note core count (`nproc`) — ktransformers CPU-expert perf scales with cores.

## Environment (shared)
- Repo: `/data/user/ypzhang/dev/github/codemk8/fastllm-py` — on **shared NFS
  `/data`**, so this is the **same working tree** the 5090 agent used (all session
  scripts + uncommitted changes are present, e.g. `scripts/bench_qwen30b_steady.py`).
- **Read the project memory first** — full context in `MEMORY.md`,
  `ktransformers-comparison.md`, `fastllm-py-project.md` (under
  `~/.claude/projects/-data-user-ypzhang-dev-github-codemk8-fastllm-py/memory/`).
- Models are **NOT shared** (5090 used local `/var/tmp`). Download Qwen3-30B-A3B
  yourself (bf16 for our INT4; GGUF Q4_K_M if you also want llama.cpp). HF CLI is
  **`hf download`** (not `huggingface-cli`).
- The repo `.venv` has cupy 14.x + our deps and works.

## Our side (fastllm-py) — target to reproduce
- 5090 result: **Qwen3-30B-A3B INT4 GPU-resident graph decode = 82.8 tok/s**. On a
  4090 expect **lower** (~half the memory bandwidth), maybe ~40-50 tok/s — that's
  fine; the same-box comparison is what matters.
- Run: `python scripts/bench_qwen30b_steady.py /path/to/Qwen3-30B-A3B` on **ONE
  4090** (our MoE graph path is single-GPU; 30B INT4 ≈ 16 GB fits in 24 GB with
  KV). It loads INT4 experts GPU-resident, captures the graph, measures
  **steady-state** decode.
- 🔑 **Methodology gotcha (critical):** measure steady-state = prime once, then
  time only `step()` replays. Do **NOT** time `generate()` end-to-end — it folds
  prefill + a graph re-capture into the per-token number and gave a bogus 19.4
  tok/s before we caught it (real answer 82.8). The steady script does it right.

## ktransformers side — the worked-out pipeline (saves you hours)
1. Isolated venv; install: `pip install torch --index-url
   https://download.pytorch.org/whl/cu124` (4090 is sm_89 — cu124/torch 2.6+ is
   fine; you do **not** need cu128), then `pip install ktransformers` (pulls
   **0.6.3 + kt-kernel** wheels) and `pip install sglang-kt`.
2. The pip wheels are **runtime-only** — quant/bench/convert scripts live in the
   git source: `git clone --depth 1 https://github.com/kvcache-ai/ktransformers.git`.
3. **INT4 weights** (fair match to ours): `kt-kernel/scripts/convert_cpu_weights.py
   --input-path <bf16 model> --input-type bf16 --output <out> --quant-method int4`
   (this step needs AVX512 — hence the gate check).
4. **Serve + measure:**
   ```
   KT_KERNEL_CPU_VARIANT=<detected> python -m sglang.launch_server \
     --host 127.0.0.1 --port 30005 \
     --model <bf16 model dir> --kt-weight-path <converted weights> \
     --kt-method AMXINT4 --kt-cpuinfer <nproc> --kt-threadpool-count 1 \
     --kt-num-gpu-experts <N> --attention-backend triton \
     --trust-remote-code --tensor-parallel-size 1 \
     --mem-fraction-static 0.90 --disable-shared-experts-fusion --disable-custom-all-reduce
   ```
   Then send an OpenAI `/v1/chat/completions` request with a large `max_tokens`
   and compute **completion_tokens / decode_time** (exclude prefill). The
   `kt bench -t inference -m <model>` CLI also exists.

## Run ktransformers in BOTH modes if the box allows — more informative
- **(a) All-experts-on-GPU INT4** (`--kt-num-gpu-experts` = all 128): matches our
  GPU-resident config → cleanest apples-to-apples.
- **(b) Native CPU-offload INT4** (their design point: most experts on CPU via
  AVX512, attention on GPU): what ktransformers is *for*, and where their
  leaderboard 52.5 came from. On 24 GB 4090s you may **need** offload if experts
  don't all fit.
- VRAM: 30B INT4 ≈ 16 GB fits one 24 GB 4090 for mode (a); if tight, use both
  4090s or offload.

## Deliverable
A table: **fastllm-py INT4 (1×4090)** vs **ktransformers INT4 mode-a / mode-b** —
decode tok/s, same model, with exact configs (GPU count, num_gpu_experts,
cpuinfer threads) and caveats. Then **append the 4090 result to the
`ktransformers-comparison` memory** so both agents' findings live together.

## What's already known (5090 box)
| | tok/s | where |
|---|---|---|
| fastllm-py INT4 (Qwen3-30B-A3B, graph) | 82.8 | 1×**5090** |
| ktransformers (Qwen3-30B-A3B **BF16** +AVX2) | 52.5 | their EPYC+5090 leaderboard |
| ktransformers on 5090 box | **can't run** | Arrow Lake CPU, no AVX512 → SIGILL |

Other gotchas learned on the 5090:
- The index-driven MoE kernel optimization was a **wash in the graph** (82.6 vs
  82.8): empty early-out blocks are ~free in a captured graph, so eager kernel
  microbenchmarks mispredict graph cost. Don't chase MoE micro-opt for speed.
- ktransformers install downgraded torch to a pinned version — verify the GPU
  still works after (`torch.cuda.get_device_capability`).

Report the 4090 numbers for **both** frameworks so we finally have a true
same-box, same-precision INT4 comparison.
