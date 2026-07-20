"""Core elementwise ops, written once and usable from NumPy or CuPy.

Correctness-first implementations. Fused CuPy RawKernel variants can be
substituted later without changing call sites.
"""
from __future__ import annotations

import numpy as np


def get_xp(x):
    return np if isinstance(x, np.ndarray) else __import__("cupy")


def rmsnorm(x, weight, eps: float):
    """x: (..., dim). Computed in fp32 like HF does."""
    xp = get_xp(x)
    xf = x.astype(xp.float32)
    var = xp.mean(xf * xf, axis=-1, keepdims=True)
    out = xf * (1.0 / xp.sqrt(var + eps))
    return (out * weight.astype(xp.float32)).astype(x.dtype)


def build_rope_cache(positions, head_dim: int, theta: float, xp=np):
    """positions: (T,) int array -> cos/sin (T, head_dim//2) float32."""
    inv_freq = 1.0 / (theta ** (xp.arange(0, head_dim, 2, dtype=xp.float32) / head_dim))
    ang = positions.astype(xp.float32)[:, None] * inv_freq[None, :]
    return xp.cos(ang), xp.sin(ang)


def yarn_get_mscale(scale: float, mscale: float) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * np.log(scale) + 1.0


def build_rope_cache_yarn(positions, head_dim: int, theta: float,
                          scaling: dict, xp=np):
    """DeepSeek-V2 YaRN rotary cache. Returns cos/sin (T, head_dim//2)
    with the YaRN attention-magnitude factor folded in."""
    factor = float(scaling["factor"])
    beta_fast = float(scaling.get("beta_fast", 32))
    beta_slow = float(scaling.get("beta_slow", 1))
    orig_max = float(scaling.get("original_max_position_embeddings", 4096))
    mscale = float(scaling.get("mscale", 1.0))
    mscale_all = float(scaling.get("mscale_all_dim", 0.0))

    def correction_dim(num_rot):
        return (head_dim * np.log(orig_max / (num_rot * 2 * np.pi))
                / (2 * np.log(theta)))

    low = max(int(np.floor(correction_dim(beta_fast))), 0)
    high = min(int(np.ceil(correction_dim(beta_slow))), head_dim - 1)
    ramp = (xp.arange(head_dim // 2, dtype=xp.float32) - low) / max(high - low, 1e-3)
    ramp = xp.clip(ramp, 0.0, 1.0)
    extra_mask = 1.0 - ramp  # 1 → extrapolate (high freq), 0 → interpolate

    exp = xp.arange(0, head_dim, 2, dtype=xp.float32) / head_dim
    freq_extra = 1.0 / (theta ** exp)
    freq_inter = 1.0 / (factor * theta ** exp)
    inv_freq = freq_inter * (1 - extra_mask) + freq_extra * extra_mask

    ang = positions.astype(xp.float32)[:, None] * inv_freq[None, :]
    m = yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all)
    return xp.cos(ang) * m, xp.sin(ang) * m


def deinterleave_rope_input(q):
    """DeepSeek stores rope dims interleaved [a0,b0,a1,b1,..] — convert to
    the half-split layout [a0,a1,..,b0,b1,..] expected by rotate_half."""
    xp = get_xp(q)
    return xp.concatenate([q[..., 0::2], q[..., 1::2]], axis=-1)


def apply_rope(q, cos, sin):
    """q: (T, heads, head_dim), HF 'rotate_half' convention:
    pairs are (x[i], x[i + dim/2])."""
    xp = get_xp(q)
    half = q.shape[-1] // 2
    qf = q.astype(xp.float32)
    q1, q2 = qf[..., :half], qf[..., half:]
    c = cos[:, None, :]
    s = sin[:, None, :]
    out = xp.concatenate([q1 * c - q2 * s, q2 * c + q1 * s], axis=-1)
    return out.astype(q.dtype)


def swiglu(gate, up):
    """silu(gate) * up, in fp32."""
    xp = get_xp(gate)
    g = gate.astype(xp.float32)
    return ((g / (1.0 + xp.exp(-g))) * up.astype(xp.float32)).astype(gate.dtype)


def softmax(x, axis=-1):
    xp = get_xp(x)
    xf = x.astype(xp.float32)
    m = xp.max(xf, axis=axis, keepdims=True)
    e = xp.exp(xf - m)
    return e / xp.sum(e, axis=axis, keepdims=True)
