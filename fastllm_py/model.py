"""Generic decoder-only transformer built from config + weight names.

Zero per-model code: features (QK-norm, attention bias, tied embeddings,
MoE layers) are detected from the HF config and from which weight names
exist in the checkpoint.

Each layer is placed on a device by DeviceMap; a layer's weights live as
CuPy arrays (cuda) or NumPy arrays (cpu) and the hidden state migrates
between devices at layer boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import ModelConfig
from .device_router import DeviceMap
from .kernels.ops import apply_rope, build_rope_cache, rmsnorm, softmax, swiglu
from .weights import WeightStore


def xp_for(device: str):
    if device.startswith("cuda"):
        import cupy as cp

        return cp
    return np


def dev_ctx(device: str):
    """Context manager making `device` current (no-op for cpu)."""
    if device.startswith("cuda"):
        import cupy as cp

        return cp.cuda.Device(int(device.split(":")[1]) if ":" in device else 0)
    import contextlib

    return contextlib.nullcontext()


def to_device(x, device: str):
    import cupy as cp

    if device.startswith("cuda"):
        dev_id = int(device.split(":")[1]) if ":" in device else 0
        with cp.cuda.Device(dev_id):
            return cp.asarray(x)
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else x


@dataclass
class KVCache:
    k: object = None  # (T, kv_heads, head_dim)
    v: object = None

    def append(self, k_new, v_new, xp):
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = xp.concatenate([self.k, k_new], axis=0)
            self.v = xp.concatenate([self.v, v_new], axis=0)
        return self.k, self.v

    @property
    def seq_len(self):
        return 0 if self.k is None else self.k.shape[0]


@dataclass
class DecoderLayer:
    idx: int
    device: str
    w: dict = field(default_factory=dict)  # name -> array on device
    is_moe: bool = False
    moe: Optional[object] = None  # MoELayer, attached in phase 2

    def has(self, name: str) -> bool:
        return name in self.w


class Model:
    def __init__(self, cfg: ModelConfig, dtype="float32"):
        self.cfg = cfg
        self.dtype = dtype
        self.layers: list[DecoderLayer] = []
        self.embed = None  # kept on the first layer's device
        self.final_norm = None
        self.lm_head = None
        self.head_device = "cpu"

    # ------------------------------------------------------------- loading
    @classmethod
    def load(cls, model_path: str, device_map: DeviceMap | None = None,
             dtype: str = "float32", moe_device: "MoeDeviceMap | None" = None,
             expert_dtype: str = "float16", gpu_cache_bytes: int = 8 << 30,
             gpu_expert_quant: str = "none") -> "Model":
        from .config import load_config

        cfg = load_config(model_path)
        store = WeightStore(model_path, bf16_to="float32")
        m = cls(cfg, dtype)
        devices = (device_map or DeviceMap({"cuda:0": 1})).assign(cfg.num_layers)

        def up(key, device):
            t = store.get(key).astype(dtype)
            return to_device(t, device)

        prefix = "model." if "model.embed_tokens.weight" in store else ""
        first_dev, last_dev = devices[0], devices[-1]
        m.embed = up(f"{prefix}embed_tokens.weight", first_dev)
        m.embed_device = first_dev

        per_layer_names = [
            "input_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.q_proj.bias",
            "self_attn.k_proj.weight", "self_attn.k_proj.bias",
            "self_attn.v_proj.weight", "self_attn.v_proj.bias",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            # MLA (DeepSeek V2/V3)
            "self_attn.q_a_proj.weight", "self_attn.q_a_layernorm.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight", "self_attn.kv_b_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        ]
        for i, dev in enumerate(devices):
            base = f"{prefix}layers.{i}."
            is_moe = cfg.is_moe_layer(i) and (base + "mlp.gate.weight" in store)
            layer = DecoderLayer(idx=i, device=dev, is_moe=is_moe)
            for name in per_layer_names:
                if base + name in store:
                    layer.w[name] = up(base + name, dev)
            m.layers.append(layer)
            if is_moe:
                m._attach_moe(layer, base, store, up, moe_device, expert_dtype,
                              gpu_cache_bytes, gpu_expert_quant)

        m.head_device = last_dev
        m.final_norm = up(f"{prefix}norm.weight", last_dev)
        if "lm_head.weight" in store and not cfg.tie_word_embeddings:
            m.lm_head = up("lm_head.weight", last_dev)
        else:
            m.lm_head = to_device(m.embed, last_dev)
        m.store = store
        return m

    def _attach_moe(self, layer: DecoderLayer, base: str, store: WeightStore,
                    up, moe_device, expert_dtype: str, gpu_cache_bytes: int,
                    gpu_expert_quant: str = "none"):
        """Build a MoELayer for this decoder layer from checkpoint names."""
        import numpy as np

        from .device_router import MoeDeviceMap
        from .expert_cache import GpuExpertCache
        from .expert_router import ExpertPlacement, SpeedEstimator
        from .moe import MoELayer

        if not hasattr(self, "_moe_shared"):
            dev_id = int(layer.device.split(":")[1]) if ":" in layer.device else 0
            self._moe_shared = {
                "cache": GpuExpertCache(gpu_cache_bytes, device=dev_id),
                "estimator": SpeedEstimator(),
                "pool": None,
            }
        moe_device = moe_device or MoeDeviceMap({"cpu": 1})

        gate_w = up(base + "mlp.gate.weight", layer.device)
        gate_bias = (up(base + "mlp.gate.bias", layer.device)
                     if base + "mlp.gate.bias" in store else None)
        e_bias = (store.get_f32(base + "mlp.gate.e_score_correction_bias")
                  if base + "mlp.gate.e_score_correction_bias" in store else None)

        experts, placement = {}, {}
        eid = 0
        while base + f"mlp.experts.{eid}.gate_proj.weight" in store:
            keys = {p: base + f"mlp.experts.{eid}.{p}_proj.weight"
                    for p in ("gate", "up", "down")}
            experts[eid] = {
                p: store.get(k).astype(expert_dtype) for p, k in keys.items()
            }
            eid += 1
        for e in range(eid):
            placement[e] = moe_device.expert_device(e, eid)

        gpu_payloads = None
        if gpu_expert_quant == "int4":
            from .moe import build_marlin_expert_payload

            gpu_payloads = {e: build_marlin_expert_payload(experts[e])
                            for e in range(eid)}

        shared = shared_gate = None
        for sname in ("mlp.shared_expert.", "mlp.shared_experts."):
            if base + sname + "gate_proj.weight" in store:
                shared = {p: up(base + f"{sname}{p}_proj.weight", layer.device)
                          for p in ("gate", "up", "down")}
                break
        if base + "mlp.shared_expert_gate.weight" in store:
            shared_gate = up(base + "mlp.shared_expert_gate.weight", layer.device)

        layer.moe = MoELayer(
            self.cfg, layer.idx, gate_w, experts,
            ExpertPlacement(placement), self._moe_shared["cache"],
            self._moe_shared["estimator"], shared_weights=shared,
            shared_gate=shared_gate, gate_bias=gate_bias, e_score_bias=e_bias,
            pool=self._moe_shared["pool"], gpu_payloads=gpu_payloads,
        )
        if self._moe_shared["pool"] is None:
            self._moe_shared["pool"] = layer.moe.pool

    # ------------------------------------------------------------- forward
    def _rope_cache(self, positions, dim, xp):
        cfg = self.cfg
        kind = (cfg.rope_scaling or {}).get("type") or (cfg.rope_scaling or {}).get("rope_type")
        if kind == "yarn":
            from .kernels.ops import build_rope_cache_yarn

            return build_rope_cache_yarn(positions, dim, cfg.rope_theta,
                                         cfg.rope_scaling, xp)
        if kind == "linear":  # position interpolation (deepseek-coder etc.)
            positions = positions / float(cfg.rope_scaling["factor"])
        return build_rope_cache(positions, dim, cfg.rope_theta, xp)

    def _sdpa(self, q, k_all, v_all, scale, xp):
        """q: (T,H,Dq), k_all: (S,H,Dq), v_all: (S,H,Dv) → (T, H*Dv) fp-x."""
        T, S = q.shape[0], k_all.shape[0]
        qf = q.astype(xp.float32).transpose(1, 0, 2)
        kf = k_all.astype(xp.float32).transpose(1, 2, 0)
        scores = (qf @ kf) * xp.float32(scale)
        q_pos = xp.arange(S - T, S)[:, None]
        mask = xp.arange(S)[None, :] > q_pos
        scores = xp.where(mask[None], xp.float32(-1e30), scores)
        probs = softmax(scores, axis=-1)
        ctx = probs @ v_all.astype(xp.float32).transpose(1, 0, 2)
        return ctx.transpose(1, 0, 2).reshape(T, -1)

    def _attention_mla(self, layer: DecoderLayer, x, positions, kv: KVCache):
        """DeepSeek V2/V3 multi-head latent attention (naive uncompressed
        cache: stores full per-head K/V like fastllm's simple path)."""
        from .kernels.ops import deinterleave_rope_input, yarn_get_mscale

        cfg = self.cfg
        xp = xp_for(layer.device)
        T = x.shape[0]
        H = cfg.num_heads
        Dn, Dr, Dv = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
        Dq = Dn + Dr

        if layer.has("self_attn.q_a_proj.weight"):  # V3 / big V2
            qa = x @ layer.w["self_attn.q_a_proj.weight"].T
            qa = rmsnorm(qa, layer.w["self_attn.q_a_layernorm.weight"], cfg.norm_eps)
            q = qa @ layer.w["self_attn.q_b_proj.weight"].T
        else:  # V2-Lite
            q = x @ layer.w["self_attn.q_proj.weight"].T
        q = q.reshape(T, H, Dq)
        q_nope, q_pe = q[..., :Dn], q[..., Dn:]

        ckv = x @ layer.w["self_attn.kv_a_proj_with_mqa.weight"].T  # (T, R+Dr)
        R = cfg.kv_lora_rank
        c_kv, k_pe = ckv[:, :R], ckv[:, R:].reshape(T, 1, Dr)
        c_kv = rmsnorm(c_kv, layer.w["self_attn.kv_a_layernorm.weight"], cfg.norm_eps)
        kv_out = (c_kv @ layer.w["self_attn.kv_b_proj.weight"].T).reshape(T, H, Dn + Dv)
        k_nope, v = kv_out[..., :Dn], kv_out[..., Dn:]

        cos, sin = self._rope_cache(positions, Dr, xp)
        q_pe = apply_rope(deinterleave_rope_input(q_pe), cos, sin)
        k_pe = apply_rope(deinterleave_rope_input(k_pe), cos, sin)
        q = xp.concatenate([q_nope, q_pe], axis=-1)

        k = xp.concatenate([k_nope, xp.broadcast_to(k_pe, (T, H, Dr))], axis=-1)
        k_all, v_all = kv.append(xp.ascontiguousarray(k), xp.ascontiguousarray(v), xp)

        scale = Dq ** -0.5
        if cfg.rope_scaling and cfg.rope_scaling.get("mscale_all_dim"):
            m = yarn_get_mscale(float(cfg.rope_scaling["factor"]),
                                float(cfg.rope_scaling["mscale_all_dim"]))
            scale *= m * m
        ctx = self._sdpa(q, k_all, v_all, scale, xp).astype(x.dtype)
        return ctx @ layer.w["self_attn.o_proj.weight"].T

    def _attention(self, layer: DecoderLayer, x, positions, kv: KVCache):
        if layer.has("self_attn.kv_a_proj_with_mqa.weight"):
            return self._attention_mla(layer, x, positions, kv)
        cfg = self.cfg
        xp = xp_for(layer.device)
        T = x.shape[0]
        H, KVH, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim

        def lin(name, inp):
            out = inp @ layer.w[f"{name}.weight"].T
            if layer.has(f"{name}.bias"):
                out = out + layer.w[f"{name}.bias"]
            return out

        q = lin("self_attn.q_proj", x).reshape(T, H, D)
        k = lin("self_attn.k_proj", x).reshape(T, KVH, D)
        v = lin("self_attn.v_proj", x).reshape(T, KVH, D)

        if layer.has("self_attn.q_norm.weight"):  # Qwen3-style per-head norm
            q = rmsnorm(q, layer.w["self_attn.q_norm.weight"], cfg.norm_eps)
            k = rmsnorm(k, layer.w["self_attn.k_norm.weight"], cfg.norm_eps)

        cos, sin = self._rope_cache(positions, D, xp)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k_all, v_all = kv.append(k, v, xp)
        rep = H // KVH
        kx = xp.repeat(k_all, rep, axis=1)  # (S, H, D)
        vx = xp.repeat(v_all, rep, axis=1)
        ctx = self._sdpa(q, kx, vx, D ** -0.5, xp).astype(x.dtype)
        return lin("self_attn.o_proj", ctx)

    def _mlp(self, layer: DecoderLayer, x):
        g = x @ layer.w["mlp.gate_proj.weight"].T
        u = x @ layer.w["mlp.up_proj.weight"].T
        return swiglu(g, u) @ layer.w["mlp.down_proj.weight"].T

    def forward(self, token_ids: np.ndarray, kv_caches: list[KVCache] | None = None):
        """token_ids: (T,) int64 for a single sequence. Returns (T, vocab) logits
        on the head device, plus updated kv caches."""
        cfg = self.cfg
        if kv_caches is None:
            kv_caches = [KVCache() for _ in range(cfg.num_layers)]
        past = kv_caches[0].seq_len
        xp0 = xp_for(self.embed_device)
        ids = to_device(np.asarray(token_ids), self.embed_device)
        with dev_ctx(self.embed_device):
            x = self.embed[ids]
        cur_dev = self.embed_device

        for layer in self.layers:
            if layer.device != cur_dev:
                x = to_device(x, layer.device)
                cur_dev = layer.device
            with dev_ctx(cur_dev):
                xp = xp_for(cur_dev)
                positions = xp.arange(past, past + x.shape[0])

                # cross-layer prefetch: warm the next MoE layer's hot experts
                # on the copy stream while this layer computes
                nxt = layer.idx + 1
                if nxt < len(self.layers) and self.layers[nxt].moe is not None:
                    self.layers[nxt].moe.prefetch_predicted()

                h = rmsnorm(x, layer.w["input_layernorm.weight"], cfg.norm_eps)
                x = x + self._attention(layer, h, positions, kv_caches[layer.idx])
                h = rmsnorm(x, layer.w["post_attention_layernorm.weight"], cfg.norm_eps)
                if layer.is_moe and layer.moe is not None:
                    x = x + layer.moe.forward(h)
                else:
                    x = x + self._mlp(layer, h)

        if cur_dev != self.head_device:
            x = to_device(x, self.head_device)
        with dev_ctx(self.head_device):
            x = rmsnorm(x, self.final_norm, cfg.norm_eps)
            logits = x @ self.lm_head.T
        return logits, kv_caches

    # ------------------------------------------------------------ generate
    def generate(self, token_ids, max_new_tokens: int = 32, temperature: float = 0.0):
        import cupy as cp

        ids = list(token_ids)
        logits, kvs = self.forward(np.asarray(ids))
        out = []
        for _ in range(max_new_tokens):
            last = logits[-1]
            if isinstance(last, cp.ndarray):
                last = cp.asnumpy(last)
            nxt = int(np.argmax(last)) if temperature == 0.0 else int(
                np.random.choice(len(last), p=np.exp((last - last.max()) / temperature)
                                 / np.exp((last - last.max()) / temperature).sum()))
            out.append(nxt)
            logits, kvs = self.forward(np.asarray([nxt]), kvs)
        return out
