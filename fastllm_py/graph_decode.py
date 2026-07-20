"""CUDA-graph-accelerated single-token decode for dense (non-MLA) models.

Validated bit-exact vs eager decode on INT4 Qwen3-0.6B, deepseek-coder-1.3b,
and R1-Distill-Qwen-1.5B, with ~4.2-4.9x decode speedup. A `verify()` gate
still checks bit-exactness at generate() time and falls back to eager if a
model ever diverges.

Root-cause history (compute-sanitizer showed 0 memory errors AND correct
output -> a race, not a memory bug): the input buffers (x/pos_idx/bias) and
the verify() KV-restore were written on the default stream while the graph
launched/read on a non-blocking self.stream, so the graph raced ahead and read
stale inputs. Fix: issue those writes on self.stream so same-stream ordering
holds. (The dedicated capture pool + per-call Marlin workspaces are also
required for correctness.)


Decode is host-dispatch-bound: each token issues ~hundreds of tiny kernel
launches whose Python/driver overhead dwarfs the GPU work. A CUDA graph
captures that whole per-token step once and replays it as a single launch.

Requirements the design satisfies:
  * one capturable (non-blocking) stream for all work;
  * static shapes — KV is preallocated to ``max_len`` and attention runs over
    the *full* buffer with an additive bias mask (unwritten slots are zeroed
    and masked to -inf), so nothing reshapes as the sequence grows;
  * static addresses — fixed input (x), position, bias, and output (logits)
    buffers are updated in place between replays;
  * no host sync inside the step — the new K/V row is written at a
    device-resident position via a RawKernel; RoPE is a gather from a
    precomputed table indexed by that same position.

Scope: dense non-MLA models on a single GPU (Qwen3, Llama/DeepSeek-LLM,
Qwen2). MLA and multi-GPU splits fall back to eager decode.
"""
from __future__ import annotations

import numpy as np

from .kernels.ops import apply_rope, build_rope_cache, rmsnorm, softmax, swiglu
from .model import matmul_w

def graph_capable(model) -> bool:
    """True if GraphDecoder supports this model: INT4 (Marlin dict) linears,
    dense (non-MoE, non-MLA), all layers on one GPU."""
    cfg = model.cfg
    if cfg.is_mla or cfg.is_moe:
        return False
    if len({l.device for l in model.layers}) != 1:
        return False
    if not next(iter({l.device for l in model.layers})).startswith("cuda"):
        return False
    return isinstance(model.layers[0].w.get("self_attn.q_proj.weight"), dict)


_WRITE_KV: dict = {}


def _write_kv_kernel(cp, ctype: str):
    """RawKernel: copy the current token's K/V rows into the cache at a
    device-resident position (capturable — index comes from a device buffer).
    Keyed by element C type so fp32 and fp16 caches both work."""
    if ctype not in _WRITE_KV:
        _WRITE_KV[ctype] = cp.RawKernel(rf"""
        #include <cuda_fp16.h>
        extern "C" __global__ void write_kv_{ctype.replace(' ', '_')}(
                {ctype}* kcache, {ctype}* vcache,
                const {ctype}* knew, const {ctype}* vnew,
                const int* pos, int row_size) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < row_size) {{
                long long p = (long long)pos[0] * row_size + i;
                kcache[p] = knew[i];
                vcache[p] = vnew[i];
            }}
        }}""", f"write_kv_{ctype.replace(' ', '_')}")
    return _WRITE_KV[ctype]


class GraphDecoder:
    def __init__(self, model, max_len: int = 2048):
        import cupy as cp

        cfg = model.cfg
        if cfg.is_mla:
            raise ValueError("GraphDecoder does not support MLA models")
        devs = {l.device for l in model.layers}
        if len(devs) != 1 or not next(iter(devs)).startswith("cuda"):
            raise ValueError("GraphDecoder requires all layers on one GPU")
        # cuBLAS can't run during CUDA-graph capture, so all linear layers
        # must be Marlin INT4 (raw-kernel launch). Load with linear_quant="int4".
        if not isinstance(model.layers[0].w.get("self_attn.q_proj.weight"), dict):
            raise ValueError("GraphDecoder requires an INT4 model "
                             "(Model.load(..., linear_quant='int4'))")

        self.cp = cp
        self.model = model
        self.cfg = cfg
        self.max_len = max_len
        self.dtype = cp.float32 if model.dtype == "float32" else cp.float16
        self.dev_id = int(next(iter(devs)).split(":")[1]) if ":" in next(iter(devs)) else 0

        H, KVH, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        hidden = cfg.hidden_dim
        with cp.cuda.Device(self.dev_id):
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.x = cp.zeros((1, hidden), dtype=self.dtype)
            self.pos_idx = cp.zeros((1,), dtype=cp.int32)
            self.bias = cp.full((max_len,), -1e30, dtype=cp.float32)
            self.hidden = cp.zeros((hidden,), dtype=self.dtype)  # graph output
            self.k_cache = [cp.zeros((max_len, KVH, D), dtype=self.dtype)
                            for _ in model.layers]
            self.v_cache = [cp.zeros((max_len, KVH, D), dtype=self.dtype)
                            for _ in model.layers]
            # full RoPE table (positions 0..max_len-1) using the model's own
            # rope semantics (handles linear position-interpolation scaling)
            cos, sin = model._rope_cache(cp.arange(max_len), D, cp)
            self.cos_tab, self.sin_tab = cos, sin  # (max_len, D/2)
        self.graph = None
        self._captured = False
        self._ws_list = []   # per-marlin-call-site workspaces (see _mm)
        self._ws_i = 0

    # ------------------------------------------------------------- one step
    def _decode_step(self):
        """Issue the full per-token decode on self.stream, reading self.x /
        self.pos_idx / self.bias and writing self.logits. Pure static shapes."""
        cp = self.cp
        cfg = self.cfg
        H, KVH, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        rep = H // KVH
        scale = D ** -0.5
        write = _write_kv_kernel(cp, "float" if self.dtype == cp.float32 else "__half")
        row = KVH * D
        blocks = (row + 127) // 128

        cos = self.cos_tab[self.pos_idx]          # (1, D/2) gather at position
        sin = self.sin_tab[self.pos_idx]
        x = self.x
        self._ws_i = 0                            # reset per-call workspace cursor
        mm = self._mm  # marlin GEMV on the capture stream (no cuBLAS)

        def lin(layer, name, inp):
            out = mm(inp, layer.w[f"{name}.weight"])
            if layer.has(f"{name}.bias"):
                out = out + layer.w[f"{name}.bias"]
            return out

        for li, layer in enumerate(self.model.layers):
            h = rmsnorm(x, layer.w["input_layernorm.weight"], cfg.norm_eps)
            q = lin(layer, "self_attn.q_proj", h).reshape(1, H, D)
            k = lin(layer, "self_attn.k_proj", h).reshape(1, KVH, D)
            v = lin(layer, "self_attn.v_proj", h).reshape(1, KVH, D)
            if layer.has("self_attn.q_norm.weight"):
                q = rmsnorm(q, layer.w["self_attn.q_norm.weight"], cfg.norm_eps)
                k = rmsnorm(k, layer.w["self_attn.k_norm.weight"], cfg.norm_eps)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            # write current K/V into the cache at pos, then attend over the
            # full buffer (unwritten slots are zeroed -> score 0 -> masked).
            # Attention is done with broadcast-multiply + reductions (NOT
            # cuBLAS matmul) so the step is CUDA-graph-capturable.
            kc, vc = self.k_cache[li], self.v_cache[li]
            write((blocks,), (128,),
                  (kc, vc, cp.ascontiguousarray(k), cp.ascontiguousarray(v),
                   self.pos_idx, np.int32(row)))
            kx = cp.repeat(kc, rep, axis=1)       # (max_len, H, D)
            vx = cp.repeat(vc, rep, axis=1)
            qh = q.reshape(H, D).astype(cp.float32)                 # (H,D)
            scores = (qh[None] * kx.astype(cp.float32)).sum(2)      # (max_len,H)
            scores = scores * cp.float32(scale) + self.bias[:, None]
            scores -= scores.max(0, keepdims=True)
            e = cp.exp(scores)
            probs = e / e.sum(0, keepdims=True)                    # (max_len,H)
            ctx = (probs[:, :, None] * vx.astype(cp.float32)).sum(0)  # (H,D)
            ctx = ctx.reshape(1, H * D).astype(self.dtype)
            x = x + lin(layer, "self_attn.o_proj", ctx)

            h = rmsnorm(x, layer.w["post_attention_layernorm.weight"], cfg.norm_eps)
            g = mm(h, layer.w["mlp.gate_proj.weight"])
            u = mm(h, layer.w["mlp.up_proj.weight"])
            x = x + mm(swiglu(g, u), layer.w["mlp.down_proj.weight"])

        # final norm inside the graph; lm_head is fp16/cuBLAS so it runs
        # OUTSIDE the graph (one matmul per token) — see _logits_from_hidden
        self.hidden[:] = rmsnorm(x, self.model.final_norm, cfg.norm_eps)[0]

    def _mm(self, inp, w):
        """Marlin INT4 GEMV on the capture stream (capturable, no cuBLAS).

        Each call site gets its OWN Marlin workspace: the module-global shared
        workspace is reused by every same-size_n GEMM, which is fine eagerly
        (stream-ordered reset) but corrupts a replayed CUDA graph. Per-site
        workspaces are allocated once during warmup and held for the graph's
        lifetime."""
        from .kernels.marlin import gemm_fast, make_workspace

        if not isinstance(w, dict):
            raise TypeError("GraphDecoder requires INT4 (marlin) linear weights")
        size_n = w["scales"].shape[1]
        i = self._ws_i
        if i >= len(self._ws_list):
            self._ws_list.append(make_workspace(size_n, self.cp))
        self._ws_i += 1
        a = inp if inp.dtype == self.cp.float16 else inp.astype(self.cp.float16)
        return gemm_fast(a, w["qweight"], w["scales"], w["zeros"],
                         size_n, inp.shape[1], stream=self.stream,
                         workspace=self._ws_list[i])

    def _logits_from_hidden(self):
        return matmul_w(self.hidden[None], self.model.lm_head)[0]

    # ------------------------------------------------------------- prefill
    def prime(self, prompt_ids):
        """Eager prefill through the normal model, then copy the resulting KV
        into the preallocated buffers and mark valid positions. Returns the
        last-token logits (numpy) and the next write position."""
        cp = self.cp
        from .model import KVCache

        logits, kvs = self.model.forward(np.asarray(prompt_ids))
        n = len(prompt_ids)
        if n > self.max_len - 1:
            raise ValueError(f"prompt {n} exceeds max_len {self.max_len}")
        with cp.cuda.Device(self.dev_id):
            # reset the mask so a reused decoder doesn't see a prior request's
            # valid slots, then mark this prompt's positions valid
            self.bias.fill(-1e30)
            for li in range(len(self.model.layers)):
                self.k_cache[li][:n] = kvs[li].k.astype(self.dtype)
                self.v_cache[li][:n] = kvs[li].v.astype(self.dtype)
            self.bias[:n] = 0.0
        last = logits[-1]
        return (cp.asnumpy(last) if isinstance(last, cp.ndarray) else last), n

    # ------------------------------------------------------------- capture
    def capture(self):
        cp = self.cp
        with cp.cuda.Device(self.dev_id):
            self.pos_idx[0] = 0
            # Capture the step's transient intermediates from a DEDICATED pool
            # that we never allocate from again. A CUDA graph bakes in the
            # addresses of its intermediates; with the shared default pool those
            # blocks are freed after capture and handed to later allocations
            # (prime, lm_head, next step), corrupting replay. Reserving them in
            # a private pool keeps the baked addresses valid forever.
            self._graph_pool = cp.cuda.MemoryPool()
            default_alloc = cp.get_default_memory_pool().malloc
            cp.cuda.set_allocator(self._graph_pool.malloc)
            try:
                # warmup so the private pool caches all intermediate blocks
                # (capture must not hit cudaMalloc)
                with self.stream:
                    for _ in range(3):
                        self._decode_step()
                self.stream.synchronize()
                with self.stream:
                    self.stream.begin_capture()
                    self._decode_step()
                self.graph = self.stream.end_capture()
            finally:
                cp.cuda.set_allocator(default_alloc)
            # warmup + capture wrote scratch into KV slot 0 — clear before use
            for li in range(len(self.model.layers)):
                self.k_cache[li].fill(0)
                self.v_cache[li].fill(0)
            self.bias.fill(-1e30)
            self.stream.synchronize()
        self._captured = True

    def _set_inputs(self, token_id: int, position: int):
        cp = self.cp
        self.x[0] = self.model.embed[token_id].astype(self.dtype)
        self.pos_idx[0] = position
        self.bias[position] = 0.0

    def step_eager(self, token_id: int, position: int):
        """Fixed-buffer decode WITHOUT graph capture (Stage-1 validation)."""
        cp = self.cp
        with cp.cuda.Device(self.dev_id), self.stream:
            self._set_inputs(token_id, position)
            self._decode_step()
            logits = self._logits_from_hidden()
        self.stream.synchronize()
        return cp.asnumpy(logits)

    def step(self, token_id: int, position: int):
        """Graph-replay decode; lm_head (cuBLAS) runs outside the graph.

        _set_inputs MUST run on self.stream: it writes the graph's input
        buffers (x/pos_idx/bias), and the graph reads them. On the default
        stream those writes race the graph launch on self.stream (non-blocking)
        -> the graph reads stale inputs. Same-stream ordering fixes it."""
        cp = self.cp
        with cp.cuda.Device(self.dev_id), self.stream:
            self._set_inputs(token_id, position)
            self.graph.launch(self.stream)
            logits = self._logits_from_hidden()
        self.stream.synchronize()
        return cp.asnumpy(logits)

    def verify(self, token_id: int, position: int, n: int = 24, atol: float = 0.0):
        """Compare graph replay vs eager _decode_step over n teacher-forced
        steps from (token_id, position). Returns True only if they agree to
        `atol` (default: BIT-EXACT). Guards against subtle capture/replay
        divergence — graph capture correctness is model-dependent, so callers
        fall back to eager unless the graph is provably identical. Runs on a
        throwaway KV snapshot so it doesn't perturb state."""
        cp = self.cp
        # original pre-verify state (restored at the end so generate() is clean)
        ok = [c.copy() for c in self.k_cache]
        ov = [c.copy() for c in self.v_cache]
        ob = self.bias.copy()
        # running reference state that advances along the eager path
        rk = [c.copy() for c in self.k_cache]
        rv = [c.copy() for c in self.v_cache]
        rb = self.bias.copy()

        def load(ks, vs, b):
            # restore on self.stream so it's ordered before the step's reads
            # (default-stream restores would race the self.stream graph launch)
            with self.stream:
                for c, s in zip(self.k_cache, ks):
                    c[...] = s
                for c, s in zip(self.v_cache, vs):
                    c[...] = s
                self.bias[...] = b
            self.stream.synchronize()

        agree = True
        tok, pos = token_id, position
        try:
            for _ in range(n):
                load(rk, rv, rb)
                lg = self.step(tok, pos)          # graph on the reference state
                load(rk, rv, rb)
                le = self.step_eager(tok, pos)     # eager on the same state
                if float(np.abs(lg - le).max()) > atol:
                    agree = False
                    break
                # advance the reference state with the eager step
                rk = [c.copy() for c in self.k_cache]
                rv = [c.copy() for c in self.v_cache]
                rb = self.bias.copy()
                tok = int(np.argmax(le))
                pos += 1
        finally:
            load(ok, ov, ob)
        return agree

    def generate(self, prompt_ids, max_new_tokens: int = 32, use_graph: bool = True,
                 verify: bool = True):
        # capture first (its warmup + KV clear must precede the real prefill)
        if use_graph and not self._captured:
            self.capture()
        first, pos = self.prime(prompt_ids)
        step = self.step if use_graph else self.step_eager
        self.graph_fellback = False
        if use_graph and verify and not self.verify(int(np.argmax(first)), pos):
            # graph replay diverges from eager for this model -> fall back so we
            # never emit wrong tokens (known capture issue on some models)
            step = self.step_eager
            self.graph_fellback = True
        out = [int(np.argmax(first))]
        for _ in range(max_new_tokens - 1):
            logits = step(out[-1], pos)
            pos += 1
            out.append(int(np.argmax(logits)))
        return out
