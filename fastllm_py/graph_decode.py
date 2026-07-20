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
  * static shapes — KV is preallocated to ``max_len``; attention is a
    flash-decode RawKernel that loops only over the valid ``[0, pos]`` keys
    (``valid_len = pos+1`` read from the device pos buffer), so its cost is
    O(valid_len), not O(max_len);
  * static addresses — fixed input/pos/output buffers updated in place;
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


def sample_logits(logits, temperature: float = 0.0, top_p: float = 1.0,
                  top_k: int = 0, rng=None) -> int:
    """Pick a token id from a (vocab,) numpy logit vector. temperature<=0 is
    greedy argmax; otherwise temperature-scale -> optional top-k -> optional
    nucleus (top-p) -> categorical sample. rng is an optional np.random.Generator
    for reproducibility (falls back to the global RNG)."""
    if temperature <= 0.0:
        return int(np.argmax(logits))
    lg = np.asarray(logits, dtype=np.float64) / temperature
    lg -= lg.max()
    probs = np.exp(lg)
    probs /= probs.sum()
    if top_k and 0 < top_k < probs.size:
        drop = np.argpartition(probs, -top_k)[:-top_k]
        probs[drop] = 0.0
        probs /= probs.sum()
    if top_p < 1.0:
        order = np.argsort(-probs)
        csum = np.cumsum(probs[order])
        cutoff = order[csum > top_p]
        if len(cutoff) > 1:            # keep the first token to cross top_p
            probs[cutoff[1:]] = 0.0
            probs /= probs.sum()
    draw = (rng or np.random).choice(probs.size, p=probs)
    return int(draw)


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
    non-MLA. MoE is supported on a single GPU when experts are INT4-resident
    (moe_device={"cuda":1} + gpu_expert_quant="int4") — the fused custom-kernel
    MoE path. Dense supports any GPU count."""
    cfg = model.cfg
    if cfg.is_mla:
        return False
    if any(not l.device.startswith("cuda") for l in model.layers):
        return False
    if not isinstance(model.layers[0].w.get("self_attn.q_proj.weight"), dict):
        return False
    if cfg.is_moe:
        if len({l.device for l in model.layers}) != 1:
            return False  # fused MoE kernel is single-GPU
        ml = next((l.moe for l in model.layers if l.moe is not None), None)
        return ml is not None and ml.gpu_payloads is not None
    return True


_ATTN_DECODE: dict = {}


def _attn_decode_kernel(cp, ctype: str):
    """Flash-decode attention RawKernel: one block per query head, D threads.
    Loops only over VALID keys [0, pos] (read from the device pos buffer), so
    its cost is O(valid_len) regardless of the KV buffer size — unlike the
    reduction path which scans the whole max_len buffer every token. GQA-aware
    (kv_head = h / (H/KVH)), online softmax, fp32 accumulation. Capturable.
    Requires head_dim D to be a power of two (holds for 64/128)."""
    key = ctype
    if key not in _ATTN_DECODE:
        _ATTN_DECODE[key] = cp.RawKernel(rf"""
        #include <cuda_fp16.h>
        extern "C" __global__ void attn_decode_{ctype.replace(' ', '_')}(
                const float* __restrict__ q,       // (H, D) fp32
                const {ctype}* __restrict__ kc,    // (max_len, KVH, D)
                const {ctype}* __restrict__ vc,    // (max_len, KVH, D)
                float* __restrict__ out,           // (H, D) fp32
                const int* __restrict__ pos,       // device scalar; valid = pos[0]+1
                int H, int KVH, int D, float scale) {{
            int h = blockIdx.x;
            int d = threadIdx.x;                   // 0..D-1
            int rep = H / KVH;
            int kvh = h / rep;
            extern __shared__ float sh[];
            float* sq = sh;                        // D
            float* red = sh + D;                   // D
            __shared__ float m, l, s_score;
            sq[d] = q[h * D + d];
            if (d == 0) {{ m = -1e30f; l = 0.0f; }}
            float acc = 0.0f;
            __syncthreads();
            int VL = pos[0] + 1;
            for (int j = 0; j < VL; j++) {{
                long long base = ((long long)j * KVH + kvh) * D;
                red[d] = sq[d] * (float)kc[base + d];
                __syncthreads();
                for (int s = D >> 1; s > 0; s >>= 1) {{
                    if (d < s) red[d] += red[d + s];
                    __syncthreads();
                }}
                if (d == 0) s_score = red[0] * scale;
                __syncthreads();
                float sc = s_score;
                float new_m = fmaxf(m, sc);
                float corr = __expf(m - new_m);
                float p = __expf(sc - new_m);
                acc = acc * corr + p * (float)vc[base + d];
                if (d == 0) {{ l = l * corr + p; m = new_m; }}
                __syncthreads();
            }}
            out[h * D + d] = acc / l;
        }}""", f"attn_decode_{ctype.replace(' ', '_')}")
    return _ATTN_DECODE[key]


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
        self._decoder = decoder
        cp = decoder.cp
        cfg = decoder.cfg
        hidden, dt = cfg.hidden_dim, decoder.dtype
        with cp.cuda.Device(dev_id):
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.x_in = cp.zeros((1, hidden), dtype=dt)      # segment input hidden
            self.pos_idx = cp.zeros((1,), dtype=cp.int32)
            self.hidden_out = cp.zeros((hidden,), dtype=dt)  # segment output hidden
        self.alloc(decoder.max_len)

    def alloc(self, max_len: int):
        """(Re)allocate the max_len-sized buffers. Attention is O(max_len) per
        token (it scans the whole bias-masked buffer), so sizing max_len to the
        actual sequence — not a fixed large default — is a big win for wide
        models. Called on construction and by GraphDecoder.resize()."""
        cp = self._decoder.cp
        cfg = self._decoder.cfg
        H, KVH, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        dt = self._decoder.dtype
        self.max_len = max_len
        self.graph = None          # invalidate any prior capture
        self.ws_list, self.ws_i = [], 0
        with cp.cuda.Device(self.dev_id):
            self.k_cache = [cp.zeros((max_len, KVH, D), dtype=dt) for _ in self.layers]
            self.v_cache = [cp.zeros((max_len, KVH, D), dtype=dt) for _ in self.layers]
            cos, sin = self._decoder.model._rope_cache(cp.arange(max_len), D, cp)
            self.cos_tab, self.sin_tab = cos, sin


class GraphDecoder:
    def __init__(self, model, max_len: int = 256):
        import cupy as cp

        cfg = model.cfg
        if cfg.is_mla:
            raise ValueError("GraphDecoder does not support MLA models")
        if not isinstance(model.layers[0].w.get("self_attn.q_proj.weight"), dict):
            raise ValueError("GraphDecoder requires an INT4 model "
                             "(Model.load(..., linear_quant='int4'))")
        if cfg.is_moe and not graph_capable(model):
            raise ValueError("GraphDecoder MoE needs INT4-resident experts "
                             "(moe_device={'cuda':1}, gpu_expert_quant='int4') on 1 GPU")

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
        if cfg.is_moe:
            self._prepare_moe()

    def _prepare_moe(self):
        """Build per-MoE-layer graph data: fp16 gate weights + row-major INT4
        stacked experts (routed + shared), plus reusable device scratch. All on
        the single GPU. Enables the capturable fused MoE decode branch."""
        cp = self.cp
        from .kernels import moe_int4

        cfg = self.cfg
        dev = self.segments[0].dev_id
        E = cfg.num_experts
        first = next(l.moe for l in self.model.layers if l.moe is not None)
        inter = first._materialize_cpu(0)["gate"].shape[0]
        sinter = first.shared["gate"].shape[0] if first.shared is not None else 0
        self._moe = {"E": E, "inter": inter, "sinter": sinter,
                     "has_shared": first.shared is not None,
                     "has_sgate": first.shared_gate is not None}
        with cp.cuda.Device(dev):
            self._moe_lbuf = cp.empty((E,), dtype=cp.float32)
            self._moe_ibuf = cp.empty((E, inter), dtype=cp.float32)
            self._moe_rw = cp.empty((E,), dtype=cp.float32)
            self._moe_out = cp.empty((cfg.hidden_dim,), dtype=cp.float32)
            self._moe_srw = cp.ones((1,), dtype=cp.float32)
            self._moe_sout = cp.empty((cfg.hidden_dim,), dtype=cp.float32)
            self._moe_sibuf = cp.empty((1, max(sinter, 1)), dtype=cp.float32)
            for layer in self.model.layers:
                ml = getattr(layer, "moe", None)
                if ml is None:
                    continue
                ews = [ml._materialize_cpu(e) for e in range(E)]
                layer._gmoe = {
                    "gate_w": ml.gate_weight.astype(cp.float16),
                    "stacked": moe_int4.build_stacked_experts(ews, 128, xp=cp),
                    "shared": (moe_int4.build_stacked_experts([ml.shared], 128, xp=cp)
                               if ml.shared is not None else None),
                    "shared_gate_w": (ml.shared_gate.astype(cp.float16)
                                      if ml.shared_gate is not None else None),
                }

    def _moe_branch(self, layer, h):
        """Capturable fused MoE FFN for one decode token. h: (1, hidden).
        Returns (hidden,) fp32: routed experts + shared expert."""
        cp = self.cp
        cfg = self.cfg
        from .kernels import moe_int4

        gm = layer._gmoe
        M = self._moe
        xr = h[0]
        # gate (custom matvec, no cuBLAS) + top-k routing (sort threshold)
        moe_int4.gate_matvec(xr, gm["gate_w"], M["E"], cfg.hidden_dim, out=self._moe_lbuf)
        e = cp.exp(self._moe_lbuf - self._moe_lbuf.max())
        probs = e / e.sum()
        K = cfg.num_experts_per_tok
        kth = cp.sort(probs)[-K]
        rw = cp.where(probs >= kth, probs, 0.0)
        if cfg.norm_topk_prob:
            rw = rw / (rw.sum() + 1e-20)
        self._moe_rw[:] = rw * cfg.routed_scaling_factor
        moe_int4.fused_moe_weighted(xr, gm["stacked"], self._moe_rw, M["E"],
                                    cfg.hidden_dim, M["inter"], out=self._moe_out,
                                    inter_buf=self._moe_ibuf)
        if gm["shared"] is not None:
            moe_int4.fused_moe_weighted(xr, gm["shared"], self._moe_srw, 1,
                                        cfg.hidden_dim, M["sinter"], out=self._moe_sout,
                                        inter_buf=self._moe_sibuf)
            if gm["shared_gate_w"] is not None:
                sg = moe_int4.gate_matvec(xr, gm["shared_gate_w"], 1, cfg.hidden_dim)
                self._moe_out += self._moe_sout * (1.0 / (1.0 + cp.exp(-sg[0])))
            else:
                self._moe_out += self._moe_sout
        return self._moe_out

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
        pos_idx and writing seg.hidden_out. Static shapes, no host sync."""
        cp = self.cp
        cfg = self.cfg
        H, KVH, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        assert (D & (D - 1)) == 0, "flash-decode kernel needs power-of-two head_dim"
        scale = D ** -0.5
        ctype = "float" if self.dtype == cp.float32 else "__half"
        write = _write_kv_kernel(cp, ctype)
        attn = _attn_decode_kernel(cp, ctype)
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
            # flash-decode attention: O(valid_len), not O(max_len)
            qh = cp.ascontiguousarray(q.reshape(H, D).astype(cp.float32))
            ctx = cp.empty((H, D), dtype=cp.float32)
            attn((H,), (D,), (qh, kc, vc, ctx, seg.pos_idx,
                              np.int32(H), np.int32(KVH), np.int32(D),
                              np.float32(scale)),
                 shared_mem=2 * D * 4)
            ctx = ctx.reshape(1, H * D).astype(self.dtype)
            x = x + lin(layer, "self_attn.o_proj", ctx)

            h = rmsnorm(x, layer.w["post_attention_layernorm.weight"], cfg.norm_eps)
            if getattr(layer, "_gmoe", None) is not None:
                x = x + self._moe_branch(layer, h)[None]
            else:
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
                for li in range(len(seg.layers)):
                    gi = seg.layer_offset + li
                    seg.k_cache[li][:n] = kvs[gi].k.astype(self.dtype)
                    seg.v_cache[li][:n] = kvs[gi].v.astype(self.dtype)
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
                seg.stream.synchronize()
        self._captured = True

    def resize(self, need: int, headroom: int = 64):
        """Ensure the graph buffers hold at least `need` positions, sizing to a
        bucket (next power of two) so growth re-captures at most log(N) times.
        Attention cost is O(max_len)/token, so keeping max_len tight matters —
        especially for wide models (the 67B is 0.5x eager at max_len=2048 but
        1.3x at max_len~=need). Re-allocates + re-captures if grown."""
        target = 1
        while target < need + headroom:
            target *= 2
        if target == self.max_len and self._captured:
            return
        self.max_len = target
        for seg in self.segments:
            seg.alloc(target)
        self._captured = False
        self.capture()

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
        """No-op with the flash-decode kernel: attention length is derived from
        the caller's `position` (valid_len = pos+1), not internal state, so
        rolling back is just passing a smaller position to the next step(). The
        stale KV rows past `keep` are simply never visited (and get overwritten).
        Kept for API compatibility (speculative decoding calls it)."""

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
                              [c.copy() for c in seg.v_cache]))
            return s

        def restore(s):
            for seg, (ks, vs) in zip(self.segments, s):
                with cp.cuda.Device(seg.dev_id), seg.stream:
                    for c, x in zip(seg.k_cache, ks):
                        c[...] = x
                    for c, x in zip(seg.v_cache, vs):
                        c[...] = x
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
                 verify: bool = True, stop_ids=None, temperature: float = 0.0,
                 top_p: float = 1.0, top_k: int = 0, seed=None):
        if use_graph:
            self.resize(len(prompt_ids) + max_new_tokens)  # tight max_len (+capture)
        elif not self._captured:
            self.capture()
        first, pos = self.prime(prompt_ids)
        step = self.step if use_graph else self.step_eager
        self.graph_fellback = False
        # verify probes the graph teacher-forced on argmax (deterministic),
        # independent of the sampling mode used for the actual rollout.
        if use_graph and verify and not self.verify(int(np.argmax(first)), pos):
            step = self.step_eager
            self.graph_fellback = True
        rng = np.random.default_rng(seed) if temperature > 0.0 else None
        pick = lambda lg: sample_logits(lg, temperature, top_p, top_k, rng)
        out = [pick(first)]
        if stop_ids and out[-1] in stop_ids:
            return out
        for _ in range(max_new_tokens - 1):
            logits = step(out[-1], pos)
            pos += 1
            out.append(pick(logits))
            if stop_ids and out[-1] in stop_ids:
                break
        return out
