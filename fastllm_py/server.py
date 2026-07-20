"""OpenAI-compatible API server.

Run:  python -m fastllm_py.server --model models/Qwen3-0.6B [--port 8000]
      [--device '{"cuda:0": 1}'] [--moe-device '{"cuda": 1, "cpu": 3}']
"""
from __future__ import annotations

import argparse
import json
import time
import uuid

from .device_router import DeviceMap, MoeDeviceMap
from .engine import AsyncEngine, ContinuousEngine, GenRequest
from .model import Model

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install fastapi uvicorn") from e

app = FastAPI(title="fastllm-py")
STATE: dict = {}


@app.get("/v1/models")
async def models():
    return {"object": "list",
            "data": [{"id": STATE["model_id"], "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(body: dict):
    tok = STATE["tokenizer"]
    engine: AsyncEngine = STATE["engine"]

    messages = body.get("messages", [])
    prompt_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True)
    if hasattr(prompt_ids, "input_ids"):  # transformers>=5 returns BatchEncoding
        prompt_ids = prompt_ids.input_ids
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    req = GenRequest(
        token_ids=prompt_ids,
        max_new_tokens=int(body.get("max_tokens") or 512),
        temperature=float(body.get("temperature") or 0.0),
        top_p=float(body.get("top_p") or 1.0),
        stop_token_ids=tuple(STATE["stop_ids"]),
    )
    await engine.submit(req)
    rid = f"chatcmpl-{uuid.uuid4().hex[:20]}"
    created = int(time.time())
    model_id = STATE["model_id"]

    if body.get("stream"):
        async def gen():
            while True:
                tid = await req.queue.get()
                if tid is None:
                    break
                if tid in req.stop_token_ids:
                    continue
                delta = tok.decode([tid])
                chunk = {"id": rid, "object": "chat.completion.chunk",
                         "created": created, "model": model_id,
                         "choices": [{"index": 0, "delta": {"content": delta},
                                      "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
            final = {"id": rid, "object": "chat.completion.chunk",
                     "created": created, "model": model_id,
                     "choices": [{"index": 0, "delta": {},
                                  "finish_reason": "stop"}]}
            yield f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    out_ids = []
    while True:
        tid = await req.queue.get()
        if tid is None:
            break
        if tid not in req.stop_token_ids:
            out_ids.append(tid)
    text = tok.decode(out_ids)
    return JSONResponse({
        "id": rid, "object": "chat.completion", "created": created,
        "model": model_id,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": req.stats.get("prompt_tokens"),
                  "completion_tokens": req.stats.get("completion_tokens"),
                  "total_tokens": (req.stats.get("prompt_tokens", 0)
                                   + req.stats.get("completion_tokens", 0)),
                  "fastllm_stats": req.stats},
    })


def main():
    import uvicorn
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--device", default='{"cuda:0": 1}')
    p.add_argument("--moe-device", default='{"cpu": 1}')
    p.add_argument("--expert-dtype", default="float16")
    p.add_argument("--gpu-cache-gb", type=float, default=8.0)
    p.add_argument("--linear-quant", default="none", choices=["none", "int4"],
                   help="quantize dense linears to INT4 (enables CUDA-graph decode)")
    p.add_argument("--no-cuda-graph", action="store_true",
                   help="disable CUDA-graph decode even for INT4 dense models")
    p.add_argument("--continuous", action="store_true",
                   help="continuous batching (many concurrent streams; greedy; "
                        "needs --linear-quant int4)")
    p.add_argument("--max-batch", type=int, default=16,
                   help="max concurrent sequences for --continuous")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = Model.load(
        args.model,
        DeviceMap(json.loads(args.device)),
        moe_device=MoeDeviceMap(json.loads(args.moe_device)),
        expert_dtype=args.expert_dtype,
        gpu_cache_bytes=int(args.gpu_cache_gb * 2**30),
        linear_quant=args.linear_quant,
    )
    stop_ids = [tok.eos_token_id] if tok.eos_token_id is not None else []
    extra = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(extra, int) and extra >= 0 and extra not in stop_ids:
        stop_ids.append(extra)

    STATE.update(model_id=args.model.rstrip("/").split("/")[-1],
                 tokenizer=tok, stop_ids=stop_ids)

    @app.on_event("startup")
    async def _start():
        if args.continuous:
            eng = ContinuousEngine(model, max_batch=args.max_batch)
            await eng.start()
            STATE["engine"] = eng
            print(f"[fastllm] decode path: continuous_batch "
                  f"(max_batch={args.max_batch})")
        else:
            eng = AsyncEngine(model, cuda_graph=not args.no_cuda_graph)
            STATE["engine"] = eng
            await eng.start()
            print(f"[fastllm] decode path: "
                  f"{'cuda_graph' if eng.gd is not None else 'eager'}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
