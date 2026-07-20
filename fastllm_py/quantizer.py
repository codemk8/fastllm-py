"""Weight quantization: FP8-E4M3 block-128 and INT4 group quantization.

Vectorized NumPy reference implementations (ports of fastllm's
FastllmQuantizeLinearWeightFP8E4M3Block128Kernel and int4 group quant).
GPU (CuPy) versions reuse the same code path — pass `xp=cupy`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FP8_E4M3_MAX = 448.0


@dataclass
class QuantizedTensor:
    kind: str  # "fp8_e4m3_block" | "int4_group"
    data: object  # uint8 array (np or cp)
    scales: object  # float array
    shape: tuple[int, int]  # original (out_features, in_features)
    block_size: int
    zeros: object = None  # int4 only


def _fp8_e4m3_encode(x, xp):
    """Encode float array -> uint8 e4m3fn bytes (saturating, no inf/nan)."""
    x = xp.clip(x, -FP8_E4M3_MAX, FP8_E4M3_MAX).astype(xp.float32)
    f32 = x.view(xp.uint32)
    sign = (f32 >> 24) & 0x80
    # round-to-nearest-even on the 20 mantissa bits being dropped
    exp_mant = (f32 & 0x7FFFFFFF).astype(xp.int64)
    # rebias exponent: fp32 bias 127 -> e4m3 bias 7; shift mantissa 23 -> 3
    # e4m3fn value = sign | eeee mmm
    out = xp.zeros(x.shape, dtype=xp.uint32)
    absx = xp.abs(x)
    # normal range for e4m3: exponent -6..8 (value >= 2^-6)
    norm = absx >= 2.0**-6
    # normals: extract exponent/mantissa with rounding
    em = exp_mant + 0x7FFFF + ((exp_mant >> 20) & 1)  # RNE at bit 20
    e = ((em >> 23) & 0xFF).astype(xp.int64) - 127 + 7
    m = (em >> 20) & 0x7
    val = (e << 3) | m
    val = xp.clip(val, 0, 0x7E)  # 0x7F is nan in e4m3fn
    out = xp.where(norm, val.astype(xp.uint32), out)
    # subnormals: value = m * 2^-9, m in 0..7
    sub = ~norm
    msub = xp.rint(absx * 512.0).astype(xp.uint32)  # 2^9
    msub = xp.clip(msub, 0, 7)
    out = xp.where(sub, msub, out)
    return (out | sign).astype(xp.uint8)


def _fp8_e4m3_decode(b, xp):
    """Decode uint8 e4m3fn -> float32."""
    b = b.astype(xp.uint32)
    sign = xp.where((b & 0x80) != 0, -1.0, 1.0).astype(xp.float32)
    e = ((b >> 3) & 0xF).astype(xp.int32)
    m = (b & 0x7).astype(xp.float32)
    normal = (2.0 ** (e - 7).astype(xp.float32)) * (1.0 + m / 8.0)
    subnorm = m * (2.0**-9)
    return sign * xp.where(e > 0, normal, subnorm)


def quantize_fp8_block(w, block_size: int = 128, xp=np) -> QuantizedTensor:
    """Per (128 x 128) 2-D block scaling, matching DeepSeek/fastllm FP8 format.

    w: (out_features, in_features) float array.
    scales: (ceil(out/bs), ceil(in/bs)) float32.
    """
    out_f, in_f = w.shape
    bs = block_size
    po, pi = (-out_f) % bs, (-in_f) % bs
    if po or pi:
        w = xp.pad(w.astype(xp.float32), ((0, po), (0, pi)))
    else:
        w = w.astype(xp.float32)
    O, I = w.shape
    blocks = w.reshape(O // bs, bs, I // bs, bs).transpose(0, 2, 1, 3)
    amax = xp.abs(blocks).max(axis=(2, 3))
    scales = xp.where(amax == 0, 1.0, amax / FP8_E4M3_MAX).astype(xp.float32)
    q = _fp8_e4m3_encode(blocks / scales[:, :, None, None], xp)
    q = q.transpose(0, 2, 1, 3).reshape(O, I)[:out_f, :in_f]
    return QuantizedTensor("fp8_e4m3_block", xp.ascontiguousarray(q), scales,
                           (out_f, in_f), bs)


def dequantize_fp8_block(qt: QuantizedTensor, xp=np):
    out_f, in_f = qt.shape
    bs = qt.block_size
    vals = _fp8_e4m3_decode(qt.data, xp)
    so = xp.repeat(qt.scales, bs, axis=0)[:out_f]
    si = xp.repeat(so, bs, axis=1)[:, :in_f]
    return vals * si


def quantize_int4_group(w, group_size: int = 128, xp=np) -> QuantizedTensor:
    """Asymmetric INT4 with per-group min/max scaling along in_features.

    Layout: two nibbles per byte, even column in low nibble.
    data: (out, in/2) uint8; scales/zeros: (out, in/group_size) float32.
    """
    out_f, in_f = w.shape
    assert in_f % group_size == 0 and in_f % 2 == 0
    g = w.astype(xp.float32).reshape(out_f, in_f // group_size, group_size)
    wmin = g.min(axis=2, keepdims=True)
    wmax = g.max(axis=2, keepdims=True)
    scale = (wmax - wmin) / 15.0
    scale = xp.where(scale == 0, 1.0, scale)
    q = xp.clip(xp.rint((g - wmin) / scale), 0, 15).astype(xp.uint8)
    q = q.reshape(out_f, in_f)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(xp.uint8)
    return QuantizedTensor(
        "int4_group", packed,
        scale[:, :, 0].astype(xp.float32), (out_f, in_f), group_size,
        zeros=wmin[:, :, 0].astype(xp.float32),
    )


def dequantize_int4_group(qt: QuantizedTensor, xp=np):
    out_f, in_f = qt.shape
    lo = (qt.data & 0xF).astype(xp.float32)
    hi = (qt.data >> 4).astype(xp.float32)
    q = xp.empty((out_f, in_f), dtype=xp.float32)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    scales = xp.repeat(qt.scales, qt.block_size, axis=1)
    zeros = xp.repeat(qt.zeros, qt.block_size, axis=1)
    return q * scales + zeros
