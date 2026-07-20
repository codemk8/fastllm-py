"""CUDA-graph-accelerated single-token decode for dense (non-MLA) models.

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
            # full RoPE table (positions 0..max_len-1)
            cos, sin = build_rope_cache(cp.arange(max_len), D, cfg.rope_theta, cp)
            self.cos_tab, self.sin_tab = cos, sin  # (max_len, D/2)
        self.graph = None
        self._captured = False

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
        """Marlin INT4 GEMV on the capture stream (capturable, no cuBLAS)."""
        from .kernels.marlin import gemm_fast

        if not isinstance(w, dict):
            raise TypeError("GraphDecoder requires INT4 (marlin) linear weights")
        a = inp if inp.dtype == self.cp.float16 else inp.astype(self.cp.float16)
        return gemm_fast(a, w["qweight"], w["scales"], w["zeros"],
                         w["scales"].shape[1], inp.shape[1], stream=self.stream)

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
            # warmup: run the step a few times so the memory pool caches all
            # intermediate blocks (capture must not hit cudaMalloc)
            with self.stream:
                for _ in range(3):
                    self._decode_step()
            self.stream.synchronize()
            # capture the step as a graph (ops must be issued on self.stream)
            with self.stream:
                self.stream.begin_capture()
                self._decode_step()
            self.graph = self.stream.end_capture()
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
        """Graph-replay decode; lm_head (cuBLAS) runs outside the graph."""
        cp = self.cp
        with cp.cuda.Device(self.dev_id):
            self._set_inputs(token_id, position)
            self.graph.launch(self.stream)
            with self.stream:
                logits = self._logits_from_hidden()
            self.stream.synchronize()
        return cp.asnumpy(logits)

    def generate(self, prompt_ids, max_new_tokens: int = 32, use_graph: bool = True):
        # capture first (its warmup + KV clear must precede the real prefill)
        if use_graph and not self._captured:
            self.capture()
        first, pos = self.prime(prompt_ids)
        step = self.step if use_graph else self.step_eager
        out = [int(np.argmax(first))]
        for _ in range(max_new_tokens - 1):
            logits = step(out[-1], pos)
            pos += 1
            out.append(int(np.argmax(logits)))
        return out
