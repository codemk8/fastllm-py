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
    def __init__(self, model: Model, cuda_graph: bool = True, graph_max_len: int = 8192,
                 draft: Model | None = None, spec_gamma: int = 4):
        self.model = model
        self.requests: asyncio.Queue[GenRequest] = asyncio.Queue()
        self._task = None
        self.graph_max_len = graph_max_len
        self.gd = None  # lazily-captured GraphDecoder (single-stream decode)
        # optional speculative decoding: a small draft proposes, target verifies.
        # Used for greedy (temperature==0) requests only; output is identical to
        # greedy target decoding. Needs a separate draft model sharing the vocab.
        self.spec = None
        if draft is not None:
            from .speculative import SpeculativeDecoder

            self.spec = SpeculativeDecoder(model, draft, gamma=spec_gamma,
                                           draft_max_len=graph_max_len)
        if cuda_graph:
            try:
                from .graph_decode import GraphDecoder, graph_capable

                if graph_capable(model):
                    gd = GraphDecoder(model, max_len=graph_max_len)
                    gd.capture()
                    # one-time bit-exact check vs eager; only keep if it holds
                    probe = np.array([1, 2, 3, 4, 5], dtype=np.int64)
                    first, pos = gd.prime(probe)
                    if gd.verify(int(np.argmax(first)), pos):
                        self.gd = gd
            except Exception:
                self.gd = None  # any capture/verify failure -> eager path

    async def start(self):
        self._task = asyncio.create_task(self._worker())

    async def submit(self, req: GenRequest) -> GenRequest:
        await self.requests.put(req)
        return req

    async def _worker(self):
        loop = asyncio.get_running_loop()
        while True:
            req = await self.requests.get()
            # Run the whole generation on ONE worker thread and stream tokens
            # back via the loop. Per-token run_in_executor round-trips would
            # otherwise dominate at graph-decode speeds (~5ms/token).
            await loop.run_in_executor(None, self._generate, req, loop)

    def _generate(self, req: GenRequest, loop):
        """Runs in a thread. Streams tokens to req.queue thread-safely."""
        import cupy as cp

        def emit(x):
            loop.call_soon_threadsafe(req.queue.put_nowait, x)

        try:
            t0 = time.time()
            ids = np.asarray(req.token_ids, dtype=np.int64)
            n_prompt = len(ids)
            produced = 0
            # speculative path: greedy only (identical to greedy target decode)
            use_spec = (self.spec is not None and req.temperature == 0.0
                        and n_prompt + req.max_new_tokens < self.graph_max_len)
            use_graph = (not use_spec and self.gd is not None
                         and n_prompt + req.max_new_tokens < self.graph_max_len)
            if use_spec:
                t_prefill = None
                state = {"n": 0}

                def _on(tok):
                    if state["n"] == 0:
                        state["t_prefill"] = time.time() - t0
                    state["n"] += 1
                    emit(tok)

                self.spec.generate(ids, max_new_tokens=req.max_new_tokens,
                                   stop_ids=req.stop_token_ids, on_token=_on)
                produced = state["n"]
                t_prefill = state.get("t_prefill", time.time() - t0)
            elif use_graph:
                last, pos = self.gd.prime(ids)
                t_prefill = time.time() - t0
                while produced < req.max_new_tokens:
                    nxt = _sample(last, req.temperature, req.top_p)
                    produced += 1
                    emit(nxt)
                    if nxt in req.stop_token_ids:
                        break
                    last = self.gd.step(nxt, pos)
                    pos += 1
            else:
                logits, kvs = self.model.forward(ids, None)
                t_prefill = time.time() - t0
                while produced < req.max_new_tokens:
                    last = logits[-1]
                    if isinstance(last, cp.ndarray):
                        last = cp.asnumpy(last)
                    nxt = _sample(last, req.temperature, req.top_p)
                    produced += 1
                    emit(nxt)
                    if nxt in req.stop_token_ids:
                        break
                    logits, kvs = self.model.forward(np.asarray([nxt]), kvs)
            dt = time.time() - t0 - t_prefill
            req.stats = {
                "prompt_tokens": n_prompt,
                "completion_tokens": produced,
                "prefill_s": round(t_prefill, 3),
                "decode_tok_s": round(produced / dt, 2) if dt > 0 else None,
                "decode_path": ("speculative" if use_spec
                                else "cuda_graph" if use_graph else "eager"),
            }
            if use_spec:
                s = self.spec.stats
                acc = s["accepted"] / s["proposed"] if s["proposed"] else 0.0
                req.stats["spec_accept"] = round(acc, 3)
                req.stats["spec_target_forwards"] = s["target_forwards"]
        except Exception as e:  # keep the worker alive; report per-request
            import traceback

            traceback.print_exc()
            req.stats["error"] = repr(e)
        finally:
            emit(None)  # sentinel
            loop.call_soon_threadsafe(req.done.set)


class ContinuousEngine:
    """Async front-end for continuous batching (fastllm_py.batched.BatchedEngine).

    A background thread runs the batching scheduler; tokens stream back to each
    request's asyncio.Queue via the event loop. Greedy decoding only (batched
    sampling is a TODO) — matches each request's single-stream greedy output.
    Drop-in for AsyncEngine's submit()/GenRequest interface, used by the server
    with --continuous. Requires a graph-capable model (INT4 dense, 1 GPU).
    """

    def __init__(self, model: Model, max_batch: int = 16, max_len: int = 4096):
        import queue as _queue

        from .batched import BatchedEngine
        from .graph_decode import graph_capable

        if not graph_capable(model):
            raise ValueError("ContinuousEngine needs a graph-capable model "
                             "(INT4 dense, single GPU)")
        self.be = BatchedEngine(model, max_batch=max_batch, max_len=max_len)
        self.incoming = _queue.Queue()
        self.loop = None
        self._thread = None

    async def start(self):
        import threading

        self.loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    async def submit(self, req: GenRequest) -> GenRequest:
        self.incoming.put(req)
        return req

    def _wire(self, req: GenRequest):
        loop = self.loop
        t0 = time.time()

        def on_token(tok):
            loop.call_soon_threadsafe(req.queue.put_nowait, tok)

        def on_done():
            req.stats = {"completion_tokens": None,
                         "decode_path": "continuous_batch"}
            loop.call_soon_threadsafe(req.queue.put_nowait, None)  # sentinel
            loop.call_soon_threadsafe(req.done.set)

        self.be.submit(req.token_ids, req.max_new_tokens, req.stop_token_ids,
                       on_token=on_token, on_done=on_done)

    def _run(self):
        import queue as _queue

        while True:
            drained = False
            try:
                while True:
                    self._wire(self.incoming.get_nowait())
                    drained = True
            except _queue.Empty:
                pass
            if self.be.step():          # ran a batch step (active work)
                continue
            if drained:                 # just submitted; loop to slot them
                continue
            # fully idle — block until the next request arrives (no busy spin)
            self._wire(self.incoming.get())
