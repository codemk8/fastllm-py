"""Async generation engine: request queue + streaming decode loop.

Single-model, single-stream decode (fastllm's target regime); requests are
served FIFO with token-level streaming. The blocking forward pass runs in a
worker thread so the event loop stays responsive.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from .model import KVCache, Model


@dataclass
class GenRequest:
    token_ids: list[int]
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    stop_token_ids: tuple[int, ...] = ()
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)  # streamed ids
    done: asyncio.Event = field(default_factory=asyncio.Event)
    stats: dict = field(default_factory=dict)


def _sample(logits: np.ndarray, temperature: float, top_p: float) -> int:
    if temperature <= 0.0:
        return int(np.argmax(logits))
    logits = logits.astype(np.float64) / temperature
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    if top_p < 1.0:
        order = np.argsort(-probs)
        csum = np.cumsum(probs[order])
        cutoff = order[csum > top_p]
        if len(cutoff) > 1:
            probs[cutoff[1:]] = 0.0
            probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


class AsyncEngine:
    def __init__(self, model: Model):
        self.model = model
        self.requests: asyncio.Queue[GenRequest] = asyncio.Queue()
        self._task = None

    async def start(self):
        self._task = asyncio.create_task(self._worker())

    async def submit(self, req: GenRequest) -> GenRequest:
        await self.requests.put(req)
        return req

    async def _worker(self):
        import cupy as cp

        loop = asyncio.get_running_loop()
        while True:
            req = await self.requests.get()
            try:
                t0 = time.time()
                kvs = None
                ids = np.asarray(req.token_ids, dtype=np.int64)
                n_prompt = len(ids)
                produced = 0
                logits, kvs = await loop.run_in_executor(
                    None, self.model.forward, ids, kvs)
                t_prefill = time.time() - t0
                while produced < req.max_new_tokens:
                    last = logits[-1]
                    if isinstance(last, cp.ndarray):
                        last = cp.asnumpy(last)
                    nxt = _sample(last, req.temperature, req.top_p)
                    produced += 1
                    await req.queue.put(nxt)
                    if nxt in req.stop_token_ids:
                        break
                    logits, kvs = await loop.run_in_executor(
                        None, self.model.forward, np.asarray([nxt]), kvs)
                dt = time.time() - t0 - t_prefill
                req.stats = {
                    "prompt_tokens": n_prompt,
                    "completion_tokens": produced,
                    "prefill_s": round(t_prefill, 3),
                    "decode_tok_s": round(produced / dt, 2) if dt > 0 else None,
                }
            except Exception as e:  # keep the worker alive; report per-request
                import traceback

                traceback.print_exc()
                req.stats["error"] = repr(e)
            finally:
                await req.queue.put(None)  # sentinel
                req.done.set()
