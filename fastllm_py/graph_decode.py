"""CUDA-graph-accelerated single-token decode for dense (non-MLA) INT4 models.

Decode is host-dispatch-bound: each token issues ~hundreds of tiny kernel
launches whose Python/driver overhead dwarfs the GPU work. A CUDA graph
captures that per-token step once and replays it as a single launch.

A CUDA graph is *per-device*, so a model whose layers are split across GPUs is
captured as **one graph per contiguous device-segment**, chained by a small
host-hopped copy of the boundary hidden state (16 KB — the 4090s have no P2P).
N=1 GPU is just the single-segment case. Per token this is N graph launches +
(N-1) tiny copies instead of ~hundreds of kernel launches.

Design invariants (per segment):
  * one capturable (non-blocking) stream on the segment's device;
  * static shapes — KV is preallocated to ``max_len`` and attention runs over
    the *full* buffer with an additive bias mask (unwritten slots are zeroed
    and masked to -inf);
  * static addresses — fixed input/pos/bias/output buffers updated in place;
  * no host sync inside the step — the new K/V row is written by a RawKernel at
    a device-resident position; RoPE is a gather from a precomputed table.

Correctness notes: linear layers must be Marlin INT4 (cuBLAS is rejected during
cupy stream capture); attention is done as broadcast-multiply + reductions (no
matmul); lm_head (fp16/cuBLAS) runs outside the graph. Input writes MUST be
issued on the capture stream (a default-stream write race was a subtle bug —
found via compute-sanitizer). Each segment uses a dedicated capture mem-pool
and per-call Marlin workspaces. `verify()` checks bit-exactness vs eager and
`generate()` falls back to eager if a model ever diverges.

Scope: dense non-MLA models (Qwen3, Llama/DeepSeek-LLM, Qwen2), any GPU count.
MLA/MoE fall back to eager decode.
"""
from __future__ import annotations

import numpy as np

from .kernels.ops import apply_rope, rmsnorm, swiglu
from .model import matmul_w


def route_gpu(logits, top_k, scoring="softmax", bias=None, n_group=0,
              topk_group=0, norm=False, scale=1.0):
    """On-GPU expert routing using ONLY capturable ops (exp/sort/where/sum —
    no cuBLAS, no host sync), so it runs inside a CUDA graph. Returns a
    (T, E) weight matrix, 0 for non-selected experts. Matches the CPU
    expert_router.route_topk selection (incl. DeepSeek V3/V4 group-limited
    routing) to float32 precision. Foundation for dense-over-experts graph
    MoE decode (compute all E experts, weight-mask to the selected top-k)."""
    import cupy as cp

    T, E = logits.shape
    if scoring == "sigmoid":
        probs = 1.0 / (1.0 + cp.exp(-logits))
        select = probs + (bias if bias is not None else 0.0)
    else:
        e = cp.exp(logits - logits.max(1, keepdims=True))
        probs = e / e.sum(1, keepdims=True)
        select = probs
    if n_group > 1 and 0 < topk_group < n_group:
        gsz = E // n_group
        gs = select.reshape(T, n_group, gsz)
        grp = cp.sort(gs, axis=-1)[:, :, -min(2, gsz):].sum(-1)
        gthr = cp.sort(grp, axis=-1)[:, -topk_group][:, None]
        select = cp.where(cp.repeat(grp >= gthr, gsz, axis=1), select, -cp.inf)
    kth = cp.sort(select, axis=-1)[:, -top_k][:, None]
    w = cp.where(select >= kth, probs, 0.0)
    if norm:
        w = w / (w.sum(1, keepdims=True) + 1e-20)
    return (w * scale).astype(cp.float32)


def graph_capable(model) -> bool:
    """True if GraphDecoder supports this model: INT4 (Marlin dict) linears,
    dense (non-MoE, non-MLA). Any number of GPUs."""
    cfg = model.cfg
    if cfg.is_mla or cfg.is_moe:
        return False
    if any(not l.device.startswith("cuda") for l in model.layers):
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


class _Segment:
    """A contiguous run of decoder layers on one GPU, plus its capture state."""

    def __init__(self, decoder, dev_id, layers, layer_offset, is_last):
        cp = decoder.cp
        cfg = decoder.cfg
        self.dev_id = dev_id
        self.layers = layers
        self.layer_offset = layer_offset  # global index of layers[0]
        self.is_last = is_last
        self.graph = None
        self.ws_list = []   # per-Marlin-call-site workspaces (see _mm)
        self.ws_i = 0
        self._pool = None

        H, KVH, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        hidden, dt = cfg.hidden_dim, decoder.dtype
        with cp.cuda.Device(dev_id):
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.x_in = cp.zeros((1, hidden), dtype=dt)      # segment input hidden
            self.pos_idx = cp.zeros((1,), dtype=cp.int32)
            self.bias = cp.full((decoder.max_len,), -1e30, dtype=cp.float32)
            self.hidden_out = cp.zeros((hidden,), dtype=dt)  # segment output hidden
            self.k_cache = [cp.zeros((decoder.max_len, KVH, D), dtype=dt)
                            for _ in layers]
            self.v_cache = [cp.zeros((decoder.max_len, KVH, D), dtype=dt)
                            for _ in layers]
            cos, sin = decoder.model._rope_cache(cp.arange(decoder.max_len), D, cp)
            self.cos_tab, self.sin_tab = cos, sin


class GraphDecoder:
    def __init__(self, model, max_len: int = 2048):
        import cupy as cp

        cfg = model.cfg
        if cfg.is_mla or cfg.is_moe:
            raise ValueError("GraphDecoder supports only dense non-MLA models")
        if not isinstance(model.layers[0].w.get("self_attn.q_proj.weight"), dict):
            raise ValueError("GraphDecoder requires an INT4 model "
                             "(Model.load(..., linear_quant='int4'))")

        self.cp = cp
        self.model = model
        self.cfg = cfg
        self.max_len = max_len
        self.dtype = cp.float32 if model.dtype == "float32" else cp.float16

        # group consecutive layers by device into segments
        self.segments = []
        cur_dev, start = model.layers[0].device, 0
        for i, layer in enumerate(model.layers + [None]):
            dev = layer.device if layer is not None else None
            if dev != cur_dev:
                seg_layers = model.layers[start:i]
                dev_id = int(cur_dev.split(":")[1]) if ":" in cur_dev else 0
                self.segments.append(
                    _Segment(self, dev_id, seg_layers, start,
                             is_last=(i == len(model.layers))))
                cur_dev, start = dev, i

        self.embed_dev = self.segments[0].dev_id
        self.head_dev = self.segments[-1].dev_id
        with cp.cuda.Device(self.head_dev):
            self.logits = cp.zeros((model.lm_head.shape[0],), dtype=self.dtype)
        self._captured = False
        self.graph_fellback = False

    # ------------------------------------------------------------ per segment
    def _mm(self, seg, inp, w):
        """Marlin INT4 GEMV on the segment's stream with a per-call-site
        workspace (a shared workspace corrupts a replayed graph)."""
        from .kernels.marlin import gemm_fast, make_workspace

        size_n = w["scales"].shape[1]
        i = seg.ws_i
        if i >= len(seg.ws_list):
            seg.ws_list.append(make_workspace(size_n, self.cp))
        seg.ws_i += 1
        a = inp if inp.dtype == self.cp.float16 else inp.astype(self.cp.float16)
        return gemm_fast(a, w["qweight"], w["scales"], w["zeros"],
                         size_n, inp.shape[1], stream=seg.stream,
                         workspace=seg.ws_list[i])

    def _segment_step(self, seg):
        """Issue this segment's layers on seg.stream, reading seg.x_in /
        pos_idx / bias and writing seg.hidden_out. Static shapes, no host sync."""
        cp = self.cp
        cfg = self.cfg
        H, KVH, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        rep = H // KVH
        scale = D ** -0.5
        write = _write_kv_kernel(cp, "float" if self.dtype == cp.float32 else "__half")
        row = KVH * D
        blocks = (row + 127) // 128
        cos = seg.cos_tab[seg.pos_idx]
        sin = seg.sin_tab[seg.pos_idx]
        x = seg.x_in
        seg.ws_i = 0

        def lin(layer, name, inp):
            out = self._mm(seg, inp, layer.w[f"{name}.weight"])
            if layer.has(f"{name}.bias"):
                out = out + layer.w[f"{name}.bias"]
            return out

        for li, layer in enumerate(seg.layers):
            h = rmsnorm(x, layer.w["input_layernorm.weight"], cfg.norm_eps)
            q = lin(layer, "self_attn.q_proj", h).reshape(1, H, D)
            k = lin(layer, "self_attn.k_proj", h).reshape(1, KVH, D)
            v = lin(layer, "self_attn.v_proj", h).reshape(1, KVH, D)
            if layer.has("self_attn.q_norm.weight"):
                q = rmsnorm(q, layer.w["self_attn.q_norm.weight"], cfg.norm_eps)
                k = rmsnorm(k, layer.w["self_attn.k_norm.weight"], cfg.norm_eps)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            kc, vc = seg.k_cache[li], seg.v_cache[li]
            write((blocks,), (128,),
                  (kc, vc, cp.ascontiguousarray(k), cp.ascontiguousarray(v),
                   seg.pos_idx, np.int32(row)))
            kx = cp.repeat(kc, rep, axis=1)
            vx = cp.repeat(vc, rep, axis=1)
            qh = q.reshape(H, D).astype(cp.float32)
            scores = (qh[None] * kx.astype(cp.float32)).sum(2)
            scores = scores * cp.float32(scale) + seg.bias[:, None]
            scores -= scores.max(0, keepdims=True)
            e = cp.exp(scores)
            probs = e / e.sum(0, keepdims=True)
            ctx = (probs[:, :, None] * vx.astype(cp.float32)).sum(0)
            ctx = ctx.reshape(1, H * D).astype(self.dtype)
            x = x + lin(layer, "self_attn.o_proj", ctx)

            h = rmsnorm(x, layer.w["post_attention_layernorm.weight"], cfg.norm_eps)
            g = self._mm(seg, h, layer.w["mlp.gate_proj.weight"])
            u = self._mm(seg, h, layer.w["mlp.up_proj.weight"])
            x = x + self._mm(seg, swiglu(g, u), layer.w["mlp.down_proj.weight"])

        if seg.is_last:  # final norm here; lm_head runs outside the graph
            seg.hidden_out[:] = rmsnorm(x, self.model.final_norm, cfg.norm_eps)[0]
        else:
            seg.hidden_out[:] = x[0]

    # ------------------------------------------------------------- prefill
    def prime(self, prompt_ids):
        """Eager prefill through the normal (possibly multi-GPU) model, then
        distribute the resulting KV into each segment's device buffers and mark
        valid positions. Returns last-token logits (numpy) + next position."""
        cp = self.cp
        logits, kvs = self.model.forward(np.asarray(prompt_ids))
        n = len(prompt_ids)
        if n > self.max_len - 1:
            raise ValueError(f"prompt {n} exceeds max_len {self.max_len}")
        for seg in self.segments:
            with cp.cuda.Device(seg.dev_id):
                seg.bias.fill(-1e30)   # reset mask (reused decoder / new request)
                for li in range(len(seg.layers)):
                    gi = seg.layer_offset + li
                    seg.k_cache[li][:n] = kvs[gi].k.astype(self.dtype)
                    seg.v_cache[li][:n] = kvs[gi].v.astype(self.dtype)
                seg.bias[:n] = 0.0
        last = logits[-1]
        return (cp.asnumpy(last) if isinstance(last, cp.ndarray) else last), n

    # ------------------------------------------------------------- capture
    def capture(self):
        cp = self.cp
        for seg in self.segments:
            with cp.cuda.Device(seg.dev_id):
                seg.pos_idx[0] = 0
                # dedicated pool: a captured graph bakes in its intermediates'
                # addresses; the shared pool would free+reuse them post-capture
                seg._pool = cp.cuda.MemoryPool()
                default_alloc = cp.get_default_memory_pool().malloc
                cp.cuda.set_allocator(seg._pool.malloc)
                try:
                    with seg.stream:
                        for _ in range(3):
                            self._segment_step(seg)
                    seg.stream.synchronize()
                    with seg.stream:
                        seg.stream.begin_capture()
                        self._segment_step(seg)
                    seg.graph = seg.stream.end_capture()
                finally:
                    cp.cuda.set_allocator(default_alloc)
                for c in seg.k_cache:
                    c.fill(0)
                for c in seg.v_cache:
                    c.fill(0)
                seg.bias.fill(-1e30)
                seg.stream.synchronize()
        self._captured = True

    # ------------------------------------------------------------- stepping
    def _run(self, token_id, position, graph: bool):
        """One decode token across all segments; returns logits (numpy).

        Segments are pipelined with CUDA events + async cross-device copies of
        the boundary hidden state — a single host sync at the very end. Per-
        segment syncs would otherwise dominate for large models (where each
        segment's GPU work is big and the dispatch savings are small)."""
        cp = self.cp
        segs = self.segments
        prev_done = None  # event: prev segment's output hidden is ready
        for si, seg in enumerate(segs):
            with cp.cuda.Device(seg.dev_id), seg.stream:
                if si == 0:
                    seg.x_in[0] = self.model.embed[token_id].astype(self.dtype)
                else:
                    # boundary hidden: prev.hidden_out (dev si-1) -> x_in (dev si),
                    # async on seg.stream, ordered after prev_done (cross-device
                    # event wait). memcpyPeerAsync works without P2P (host-staged).
                    seg.stream.wait_event(prev_done)
                    prev = segs[si - 1]
                    nbytes = seg.x_in.size * seg.x_in.itemsize
                    cp.cuda.runtime.memcpyPeerAsync(
                        seg.x_in.data.ptr, seg.dev_id,
                        prev.hidden_out.data.ptr, prev.dev_id,
                        nbytes, seg.stream.ptr)
                seg.pos_idx[0] = position
                seg.bias[position] = 0.0
                if graph:
                    seg.graph.launch(seg.stream)
                else:
                    self._segment_step(seg)
                prev_done = cp.cuda.Event()
                prev_done.record(seg.stream)
        last = segs[-1]
        with cp.cuda.Device(last.dev_id), last.stream:
            logits = matmul_w(last.hidden_out[None], self.model.lm_head)[0]
        last.stream.synchronize()  # the one and only host sync per token
        return cp.asnumpy(logits)

    def truncate(self, keep: int):
        """Roll the KV back to `keep` valid positions by re-masking the rest
        (the cached rows past `keep` become stale but masked, and are
        overwritten by the next step). Used by speculative decoding."""
        cp = self.cp
        for seg in self.segments:
            with cp.cuda.Device(seg.dev_id), seg.stream:
                seg.bias[keep:] = -1e30
            seg.stream.synchronize()

    def step(self, token_id, position):
        return self._run(token_id, position, graph=True)

    def step_eager(self, token_id, position):
        return self._run(token_id, position, graph=False)

    # ------------------------------------------------------------- verify
    def verify(self, token_id, position, n=24, atol=0.0):
        """Bit-exact graph-vs-eager over n teacher-forced steps (on a throwaway
        KV snapshot). Returns True if they agree everywhere."""
        cp = self.cp

        def snap():
            s = []
            for seg in self.segments:
                with cp.cuda.Device(seg.dev_id):
                    s.append(([c.copy() for c in seg.k_cache],
                              [c.copy() for c in seg.v_cache], seg.bias.copy()))
            return s

        def restore(s):
            for seg, (ks, vs, b) in zip(self.segments, s):
                with cp.cuda.Device(seg.dev_id), seg.stream:
                    for c, x in zip(seg.k_cache, ks):
                        c[...] = x
                    for c, x in zip(seg.v_cache, vs):
                        c[...] = x
                    seg.bias[...] = b
                seg.stream.synchronize()

        orig, ref = snap(), snap()
        agree, tok, pos = True, token_id, position
        try:
            for _ in range(n):
                restore(ref)
                lg = self.step(tok, pos)
                restore(ref)
                le = self.step_eager(tok, pos)
                if float(np.abs(lg - le).max()) > atol:
                    agree = False
                    break
                ref = snap()
                tok = int(np.argmax(le))
                pos += 1
        finally:
            restore(orig)
        return agree

    # ------------------------------------------------------------- generate
    def generate(self, prompt_ids, max_new_tokens: int = 32, use_graph: bool = True,
                 verify: bool = True):
        if use_graph and not self._captured:
            self.capture()
        first, pos = self.prime(prompt_ids)
        step = self.step if use_graph else self.step_eager
        self.graph_fellback = False
        if use_graph and verify and not self.verify(int(np.argmax(first)), pos):
            step = self.step_eager
            self.graph_fellback = True
        out = [int(np.argmax(first))]
        for _ in range(max_new_tokens - 1):
            logits = step(out[-1], pos)
            pos += 1
            out.append(int(np.argmax(logits)))
        return out
