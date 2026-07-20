"""Parse a HuggingFace config.json into a model-agnostic graph description.

No per-model code: everything the forward pass needs is derived from config
keys (plus, at load time, from which weight names actually exist).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Architecture hyper-parameters shared by decoder-only transformers."""

    model_type: str
    num_layers: int
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_dim: int
    vocab_size: int
    rope_theta: float
    norm_eps: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    # --- MoE (0 / None for dense models) ---
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_dim: int = 0
    num_shared_experts: int = 0
    first_k_dense_replace: int = 0  # DeepSeek: first k layers are dense
    moe_layer_freq: int = 1
    norm_topk_prob: bool = False
    routed_scaling_factor: float = 1.0
    scoring_func: str = "softmax"  # softmax | sigmoid
    n_group: int = 0
    topk_group: int = 0
    # --- MLA (DeepSeek V2/V3; 0/None = standard attention) ---
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    rope_scaling: dict | None = None
    # --- misc ---
    hidden_act: str = "silu"
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank > 0

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    def is_moe_layer(self, layer_idx: int) -> bool:
        if not self.is_moe:
            return False
        if layer_idx < self.first_k_dense_replace:
            return False
        return (layer_idx % self.moe_layer_freq) == 0


def _get(cfg: dict, *names, default=None):
    for n in names:
        if n in cfg and cfg[n] is not None:
            return cfg[n]
    return default


def load_config(model_path: str | Path) -> ModelConfig:
    """Read <model_path>/config.json and map heterogeneous HF key names."""
    path = Path(model_path) / "config.json"
    cfg = json.loads(path.read_text())
    # Some models nest the text config (e.g. multimodal wrappers)
    if "text_config" in cfg:
        merged = dict(cfg)
        merged.update(cfg["text_config"])
        cfg = merged

    hidden = _get(cfg, "hidden_size", "n_embd")
    heads = _get(cfg, "num_attention_heads", "n_head")
    head_dim = _get(cfg, "head_dim", default=hidden // heads)

    return ModelConfig(
        model_type=_get(cfg, "model_type", default="unknown"),
        num_layers=_get(cfg, "num_hidden_layers", "n_layer"),
        hidden_dim=hidden,
        num_heads=heads,
        num_kv_heads=_get(cfg, "num_key_value_heads", default=heads),
        head_dim=head_dim,
        intermediate_dim=_get(cfg, "intermediate_size", default=4 * hidden),
        vocab_size=_get(cfg, "vocab_size"),
        rope_theta=float(_get(cfg, "rope_theta", default=10000.0)),
        norm_eps=float(_get(cfg, "rms_norm_eps", "layer_norm_epsilon", default=1e-6)),
        max_position_embeddings=_get(cfg, "max_position_embeddings", default=32768),
        tie_word_embeddings=bool(_get(cfg, "tie_word_embeddings", default=False)),
        attention_bias=bool(_get(cfg, "attention_bias", "qkv_bias", default=False)),
        num_experts=_get(cfg, "num_experts", "n_routed_experts", "num_local_experts", default=0) or 0,
        num_experts_per_tok=_get(cfg, "num_experts_per_tok", "num_experts_per_token", default=0) or 0,
        moe_intermediate_dim=_get(cfg, "moe_intermediate_size", default=0) or 0,
        num_shared_experts=_get(cfg, "n_shared_experts", default=0) or 0,
        first_k_dense_replace=_get(cfg, "first_k_dense_replace", default=0) or 0,
        moe_layer_freq=_get(cfg, "moe_layer_freq", "decoder_sparse_step", default=1) or 1,
        norm_topk_prob=bool(_get(cfg, "norm_topk_prob", default=False)),
        routed_scaling_factor=float(_get(cfg, "routed_scaling_factor", default=1.0)),
        scoring_func=_get(cfg, "scoring_func", default="softmax"),
        n_group=_get(cfg, "n_group", default=0) or 0,
        topk_group=_get(cfg, "topk_group", default=0) or 0,
        q_lora_rank=_get(cfg, "q_lora_rank", default=0) or 0,
        kv_lora_rank=_get(cfg, "kv_lora_rank", default=0) or 0,
        qk_nope_head_dim=_get(cfg, "qk_nope_head_dim", default=0) or 0,
        qk_rope_head_dim=_get(cfg, "qk_rope_head_dim", default=0) or 0,
        v_head_dim=_get(cfg, "v_head_dim", default=0) or 0,
        rope_scaling=_get(cfg, "rope_scaling", default=None),
        hidden_act=_get(cfg, "hidden_act", default="silu"),
        raw=cfg,
    )
