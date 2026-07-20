"""Batched single-token decode for INT4 dense models.

Decode is bottlenecked on reading the (INT4) weights from VRAM once per token.
Running B sequences together reads each weight ONCE for all B and does B× the
tiny GEMV work — so aggregate tok/s scales with batch until compute-bound. This
is the throughput lever for serving many concurrent streams (e.g. subagents).

Scope (increment 1): dense non-MLA INT4, single GPU, eager (no graph yet). Each
sequence has its own KV; the batched flash-decode kernel loops only over each
sequence's own valid keys. Correctness target: each sequence's output is
identical to single-stream `Model.generate` for the same prompt.
"""
from __future__ import annotations

import numpy as np

from .kernels.ops import rmsnorm, swiglu
from .model import matmul_w

_BWRITE_KV: dict = {}
_BATTN: dict = {}


def _bwrite_kv(cp, ctype):
    if ctype not in _BWRITE_KV:
        _BWRITE_KV[ctype] = cp.RawKernel(rf"""
        #include <cuda_fp16.h>
        extern "C" __global__ void bwrite_kv_{ctype.replace(' ', '_')}(
                {ctype}* kc, {ctype}* vc, const {ctype}* kn, const {ctype}* vn,
                const int* pos, int B, int max_len, int row) {{
            // grid.x over row elements, grid.y over batch
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int b = blockIdx.y;
            if (i < row && b < B) {{
                long long dst = ((long long)b * max_len + pos[b]) * row + i;
                long long src = (long long)b * row + i;
                kc[dst] = kn[src];
                vc[dst] = vn[src];
            }}
        }}""", f"bwrite_kv_{ctype.replace(' ', '_')}")
    return _BWRITE_KV[ctype]


def _battn(cp, ctype):
    """Batched flash-decode: one block per (batch b, query head h), D threads.
    Each sequence attends to its own KV[b, 0..pos[b]]. Online softmax, GQA."""
    if ctype not in _BATTN:
        _BATTN[ctype] = cp.RawKernel(rf"""
        #include <cuda_fp16.h>
        extern "C" __global__ void battn_{ctype.replace(' ', '_')}(
                const float* __restrict__ q,     // (B, H, D)
                const {ctype}* __restrict__ kc,  // (B, max_len, KVH, D)
                const {ctype}* __restrict__ vc,  // (B, max_len, KVH, D)
                float* __restrict__ out,         // (B, H, D)
                const int* __restrict__ pos,     // (B,)  valid = pos[b]+1
                int B, int H, int KVH, int D, int max_len, float scale) {{
            int b = blockIdx.y;
            int h = blockIdx.x;
            int d = threadIdx.x;
            if (b >= B) return;
            int rep = H / KVH;
            int kvh = h / rep;
            extern __shared__ float sh[];
            float* sq = sh; float* red = sh + D;
            __shared__ float m, l, s_score;
            long long qbase = ((long long)b * H + h) * D;
            sq[d] = q[qbase + d];
            if (d == 0) {{ m = -1e30f; l = 0.0f; }}
            float acc = 0.0f;
            __syncthreads();
            int VL = pos[b] + 1;
            long long kvseq = (long long)b * max_len * KVH * D;
            for (int j = 0; j < VL; j++) {{
                long long base = kvseq + ((long long)j * KVH + kvh) * D;
                red[d] = sq[d] * (float)kc[base + d];
                __syncthreads();
                for (int s = D >> 1; s > 0; s >>= 1) {{
                    if (d < s) red[d] += red[d + s];
                    __syncthreads();
                }}
                if (d == 0) s_score = red[0] * scale;
                __syncthreads();
                float sc = s_score;
                float nm = fmaxf(m, sc);
                float corr = __expf(m - nm);
                float p = __expf(sc - nm);
                acc = acc * corr + p * (float)vc[base + d];
                if (d == 0) {{ l = l * corr + p; m = nm; }}
                __syncthreads();
            }}
            out[qbase + d] = acc / l;
        }}""", f"battn_{ctype.replace(' ', '_')}")
    return _BATTN[ctype]


class BatchedDecoder:
    def __init__(self, model, batch_size: int, max_len: int = 2048):
        import cupy as cp

        cfg = model.cfg
        if cfg.is_mla or cfg.is_moe:
            raise ValueError("BatchedDecoder: dense non-MLA only")
        if not isinstance(model.layers[0].w.get("self_attn.q_proj.weight"), dict):
            raise ValueError("BatchedDecoder requires an INT4 model")
        if len({l.device for l in model.layers}) != 1:
            raise ValueError("BatchedDecoder: single GPU only")
        self.cp = cp
        self.model = model
        self.cfg = cfg
        self.B = batch_size
        self.max_len = max_len
        self.dtype = cp.float32 if model.dtype == "float32" else cp.float16
        self.dev = int(next(iter({l.device for l in model.layers})).split(":")[1]
                       ) if ":" in model.layers[0].device else 0
        H, KVH, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        with cp.cuda.Device(self.dev):
            self.pos = cp.zeros((batch_size,), dtype=cp.int32)
            self.k_cache = [cp.zeros((batch_size, max_len, KVH, D), dtype=self.dtype)
                            for _ in model.layers]
            self.v_cache = [cp.zeros((batch_size, max_len, KVH, D), dtype=self.dtype)
                            for _ in model.layers]
            self.cos_tab, self.sin_tab = model._rope_cache(cp.arange(max_len), D, cp)
            # graph-capture state (increment 2)
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.tok_buf = cp.zeros((batch_size,), dtype=cp.int32)   # graph input
            self.hidden_out = cp.zeros((batch_size, cfg.hidden_dim), dtype=self.dtype)
        self.graph = None
        self._pool = None
        self.ws_list, self.ws_i = [], 0

    def prime(self, prompt_list):
        """Prefill each sequence (eager, single-stream) and load its KV into the
        batched buffers. prompt_list: list of B token-id lists (varied lengths).
        Returns the first token per sequence (B,) int."""
        cp = self.cp
        assert len(prompt_list) == self.B
        first = np.empty(self.B, dtype=np.int64)
        with cp.cuda.Device(self.dev):
            for b, ids in enumerate(prompt_list):
                logits, kvs = self.model.forward(np.asarray(ids, dtype=np.int64))
                n = len(ids)
                for li in range(len(self.model.layers)):
                    self.k_cache[li][b, :n] = kvs[li].k.astype(self.dtype)
                    self.v_cache[li][b, :n] = kvs[li].v.astype(self.dtype)
                self.pos[b] = n - 1  # last written position
                row = cp.asnumpy(logits[-1]) if isinstance(logits, cp.ndarray) else logits[-1]
                first[b] = int(np.argmax(row))
        return first

    def _mm(self, inp, w):
        """Batched marlin GEMV (M=B) on self.stream with a per-call workspace
        (shared workspace corrupts a replayed graph)."""
        from .kernels.marlin import gemm_fast, make_workspace

        size_n = w["scales"].shape[1]
        i = self.ws_i
        if i >= len(self.ws_list):
            self.ws_list.append(make_workspace(size_n, self.cp))
        self.ws_i += 1
        a = inp if inp.dtype == self.cp.float16 else inp.astype(self.cp.float16)
        return gemm_fast(a, w["qweight"], w["scales"], w["zeros"], size_n,
                         inp.shape[1], stream=self.stream, workspace=self.ws_list[i])

    def _step_core(self):
        """The whole batched decode step on self.stream, reading self.tok_buf /
        self.pos and writing self.hidden_out. Increments pos (so a captured
        graph auto-advances on replay). Static shapes; capturable."""
        cp = self.cp
        cfg = self.cfg
        H, KVH, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        scale = D ** -0.5
        ctype = "float" if self.dtype == cp.float32 else "__half"
        wk, at = _bwrite_kv(cp, ctype), _battn(cp, ctype)
        row = KVH * D
        self.ws_i = 0
        self.pos += 1
        posn = self.pos
        x = self.model.embed[self.tok_buf].astype(self.dtype)   # (B, hidden)
        cos = self.cos_tab[posn]
        sin = self.sin_tab[posn]

        def lin(name, inp, w):
            o = self._mm(inp, w[f"{name}.weight"])
            if f"{name}.bias" in w:
                o = o + w[f"{name}.bias"]
            return o

        for li, layer in enumerate(self.model.layers):
            w = layer.w
            h = rmsnorm(x, w["input_layernorm.weight"], cfg.norm_eps)
            q = lin("self_attn.q_proj", h, w).reshape(self.B, H, D)
            k = lin("self_attn.k_proj", h, w).reshape(self.B, KVH, D)
            v = lin("self_attn.v_proj", h, w).reshape(self.B, KVH, D)
            if "self_attn.q_norm.weight" in w:
                q = rmsnorm(q, w["self_attn.q_norm.weight"], cfg.norm_eps)
                k = rmsnorm(k, w["self_attn.k_norm.weight"], cfg.norm_eps)
            q = self._rope(q, cos, sin, H, D)
            k = self._rope(k, cos, sin, KVH, D)

            kc, vc = self.k_cache[li], self.v_cache[li]
            wk(((row + 127) // 128, self.B), (128,),
               (kc, vc, cp.ascontiguousarray(k), cp.ascontiguousarray(v),
                posn, np.int32(self.B), np.int32(self.max_len), np.int32(row)))
            qf = cp.ascontiguousarray(q.astype(cp.float32))
            ctx = cp.empty((self.B, H, D), dtype=cp.float32)
            at((H, self.B), (D,),
               (qf, kc, vc, ctx, posn, np.int32(self.B), np.int32(H),
                np.int32(KVH), np.int32(D), np.int32(self.max_len), np.float32(scale)),
               shared_mem=2 * D * 4)
            ctx = ctx.reshape(self.B, H * D).astype(self.dtype)
            x = x + lin("self_attn.o_proj", ctx, w)

            h = rmsnorm(x, w["post_attention_layernorm.weight"], cfg.norm_eps)
            g = self._mm(h, w["mlp.gate_proj.weight"])
            u = self._mm(h, w["mlp.up_proj.weight"])
            x = x + self._mm(swiglu(g, u), w["mlp.down_proj.weight"])

        self.hidden_out[:] = rmsnorm(x, self.model.final_norm, cfg.norm_eps)

    def step(self, tokens):
        """Eager batched step. tokens: (B,) int. Returns logits (B, vocab) np."""
        cp = self.cp
        with cp.cuda.Device(self.dev), self.stream:
            self.tok_buf[:] = cp.asarray(tokens, dtype=cp.int32)
            self._step_core()
            logits = matmul_w(self.hidden_out, self.model.lm_head)
        self.stream.synchronize()
        return cp.asnumpy(logits)

    def capture(self):
        """Capture _step_core as a CUDA graph (dedicated pool + on-stream inputs,
        same requirements as GraphDecoder)."""
        cp = self.cp
        with cp.cuda.Device(self.dev):
            self._pool = cp.cuda.MemoryPool()
            default_alloc = cp.get_default_memory_pool().malloc
            cp.cuda.set_allocator(self._pool.malloc)
            base = self.pos.copy()
            try:
                with self.stream:
                    for _ in range(3):
                        self.pos[:] = base          # warmup shouldn't advance state
                        self._step_core()
                self.stream.synchronize()
                self.pos[:] = base
                with self.stream:
                    self.stream.begin_capture()
                    self._step_core()
                self.graph = self.stream.end_capture()
            finally:
                cp.cuda.set_allocator(default_alloc)
            # warmup/capture wrote scratch K/V at base+1 — clear before real use
            for c in self.k_cache:
                c.fill(0)
            for c in self.v_cache:
                c.fill(0)
            self.pos[:] = base
            self.stream.synchronize()

    def step_graph(self, tokens, idle_mask=None):
        """Graph-replay batched step; lm_head (cuBLAS) runs outside the graph.

        idle_mask (bool, len B): slots with no active sequence. The graph
        increments every slot's position, so idle slots are pinned to pos -1
        here (→ 0 after the graph's +1) to keep their attention bounded to one
        (garbage, ignored) key instead of growing without limit."""
        cp = self.cp
        with cp.cuda.Device(self.dev), self.stream:
            if idle_mask is not None and idle_mask.any():
                self.pos[cp.asarray(idle_mask)] = -1
            self.tok_buf[:] = cp.asarray(tokens, dtype=cp.int32)
            self.graph.launch(self.stream)
            logits = matmul_w(self.hidden_out, self.model.lm_head)
        self.stream.synchronize()
        return cp.asnumpy(logits)

    def set_slot(self, slot, kvs, n):
        """Load one sequence's prefill KV into a slot and set its position, for
        continuous batching (reuse a captured decoder across requests). Runs on
        self.stream so the writes are ordered before the next graph step."""
        cp = self.cp
        with cp.cuda.Device(self.dev), self.stream:
            for li in range(len(self.model.layers)):
                self.k_cache[li][slot, :n] = kvs[li].k.astype(self.dtype)
                self.v_cache[li][slot, :n] = kvs[li].v.astype(self.dtype)
            self.pos[slot] = n - 1
        self.stream.synchronize()

    def _rope(self, t, cos, sin, nheads, D):
        """t: (B, nheads, D); cos/sin: (B, D/2) -> half-split rotate."""
        cp = self.cp
        half = D // 2
        tf = t.astype(cp.float32)
        t1, t2 = tf[..., :half], tf[..., half:]
        c = cos[:, None, :]
        s = sin[:, None, :]
        out = cp.concatenate([t1 * c - t2 * s, t2 * c + t1 * s], axis=-1)
        return out.astype(t.dtype)

    def generate(self, prompt_list, max_new_tokens: int = 32, use_graph: bool = False):
        step = self.step
        if use_graph:
            if self.graph is None:
                self.capture()   # before prime: warmup/clear must precede real KV
            step = self.step_graph
        first = self.prime(prompt_list)
        outs = [[int(t)] for t in first]
        cur = first.copy()
        for _ in range(max_new_tokens - 1):
            logits = step(cur)
            cur = logits.argmax(-1).astype(np.int64)
            for b in range(self.B):
                outs[b].append(int(cur[b]))
        return outs


class _Req:
    __slots__ = ("ids", "max_new", "stop", "on_token", "on_done", "produced")

    def __init__(self, ids, max_new, stop, on_token, on_done):
        self.ids = list(ids)
        self.max_new = max_new
        self.stop = set(stop or ())
        self.on_token = on_token   # callable(token_id)
        self.on_done = on_done     # callable()
        self.produced = 0


class BatchedEngine:
    """Continuous-batching scheduler over a captured BatchedDecoder.

    A fixed pool of `max_batch` slots; each step runs the batched graph over all
    slots (idle ones pinned + ignored). When a sequence finishes (EOS or
    max_new_tokens) its slot frees and the next queued request is prefilled into
    it — so sequences join and leave without re-capturing. Greedy decoding;
    each request's output is identical to its single-stream generation.

    Drive it by calling submit() then step() in a loop until idle (a background
    thread wraps this for the async server).
    """

    def __init__(self, model, max_batch: int = 16, max_len: int = 2048):
        self.model = model
        self.B = max_batch
        self.bd = BatchedDecoder(model, max_batch, max_len)
        self.bd.capture()
        self.slots: list = [None] * max_batch      # slot -> _Req or None
        self.cur = np.zeros(max_batch, dtype=np.int64)
        from collections import deque
        self.queue = deque()

    def submit(self, ids, max_new_tokens=256, stop_ids=(), on_token=None, on_done=None):
        self.queue.append(_Req(ids, max_new_tokens, stop_ids, on_token, on_done))

    def _emit(self, slot, req, tok):
        req.produced += 1
        if req.on_token:
            req.on_token(tok)
        if tok in req.stop or req.produced >= req.max_new:
            if req.on_done:
                req.on_done()
            self.slots[slot] = None

    def _fill_idle(self):
        for slot in range(self.B):
            if self.slots[slot] is None and self.queue:
                req = self.queue.popleft()
                logits, kvs = self.model.forward(np.asarray(req.ids, dtype=np.int64))
                self.bd.set_slot(slot, kvs, len(req.ids))
                cp = self.bd.cp
                row = cp.asnumpy(logits[-1]) if isinstance(logits, cp.ndarray) else logits[-1]
                first = int(np.argmax(row))
                self.cur[slot] = first
                self.slots[slot] = req
                self._emit(slot, req, first)   # emit the prefill's first token

    def step(self) -> bool:
        """Advance all active slots one token. Returns True if any work ran."""
        self._fill_idle()
        active = [s for s in range(self.B) if self.slots[s] is not None]
        if not active:
            return False
        idle_mask = np.array([self.slots[s] is None for s in range(self.B)])
        logits = self.bd.step_graph(self.cur, idle_mask=idle_mask)  # (B, vocab)
        for slot in active:
            req = self.slots[slot]
            if req is None:      # finished during prefill emit this round
                continue
            tok = int(logits[slot].argmax())
            self.cur[slot] = tok
            self._emit(slot, req, tok)
        return True

    def run_until_idle(self, max_steps: int = 100000):
        steps = 0
        while (any(s is not None for s in self.slots) or self.queue) and steps < max_steps:
            if not self.step():
                break
            steps += 1
