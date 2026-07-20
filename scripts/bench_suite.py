#!/usr/bin/env python
"""Run the benchmark suite: one subprocess per model config (clean VRAM/RAM).

Usage: bench_suite.py [suite.json] [--only NAME] [--out benchmarks/]
Appends to benchmarks/results.jsonl and regenerates benchmarks/RESULTS.md.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_one(cfg: dict, timeout: int = 3600) -> dict:
    cmd = [sys.executable, str(ROOT / "scripts" / "bench_one.py"), json.dumps(cfg)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=ROOT)
    except subprocess.TimeoutExpired:
        return {"name": cfg["name"], "variant": cfg.get("variant", "default"),
                "error": f"timeout after {timeout}s"}
    for line in proc.stdout.splitlines():
        if line.startswith("BENCH_RESULT "):
            return json.loads(line[len("BENCH_RESULT "):])
    return {"name": cfg["name"], "variant": cfg.get("variant", "default"),
            "error": (proc.stderr or proc.stdout)[-800:],
            "wall_s": round(time.time() - t0, 1)}


def write_markdown(results_path: Path, md_path: Path):
    # keep only the latest entry per (name, variant)
    latest = {}
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        latest[(r["name"], r.get("variant", "default"))] = r
    rows = sorted(latest.values(), key=lambda r: (r["name"], r.get("variant", "")))

    lines = [
        "# Benchmark results",
        "",
        f"_Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}; "
        "single-stream greedy decode, 128-token prefill, 64 decode steps; "
        "dev box: RTX 4090 (24 GB), NFS storage._",
        "",
        "| Model | Variant | Arch | Load s | Prefill tok/s | Decode tok/s (p50) "
        "| GPU GB | Cache hit |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['name']} | {r.get('variant','')} | — | — | — | "
                         f"ERROR (see results.jsonl) | — | — |")
            continue
        lines.append(
            f"| {r['name']} | {r.get('variant','default')} | {r['params_note']} "
            f"| {r['load_s']} | {r['prefill_tok_s']} "
            f"| {r['decode_tok_s']} ({r['decode_p50']}) "
            f"| {r['gpu_mem_gb']} | {r.get('cache_hit', '—')} |")
    md_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", nargs="?", default=str(ROOT / "benchmarks" / "suite.json"))
    ap.add_argument("--only", default=None, help="run only configs whose name contains this")
    ap.add_argument("--out", default=str(ROOT / "benchmarks"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    results_path = out / "results.jsonl"
    suite = json.loads(Path(args.suite).read_text())

    git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    for cfg in suite:
        if args.only and args.only.lower() not in cfg["name"].lower():
            continue
        if not (ROOT / cfg["path"]).exists():
            print(f"skip {cfg['name']} ({cfg.get('variant','default')}): "
                  f"{cfg['path']} not downloaded")
            continue
        label = f"{cfg['name']} [{cfg.get('variant', 'default')}]"
        print(f"=== {label} ...", flush=True)
        r = run_one(cfg)
        r["commit"] = git_rev
        r["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with results_path.open("a") as f:
            f.write(json.dumps(r) + "\n")
        if "error" in r:
            print(f"    ERROR: {r['error'][:300]}")
        else:
            print(f"    load {r['load_s']}s | prefill {r['prefill_tok_s']} tok/s | "
                  f"decode {r['decode_tok_s']} tok/s | gpu {r['gpu_mem_gb']}GB"
                  + (f" | hit {r['cache_hit']:.0%}" if "cache_hit" in r else ""))
        write_markdown(results_path, out / "RESULTS.md")
    print("done ->", out / "RESULTS.md")


if __name__ == "__main__":
    main()
