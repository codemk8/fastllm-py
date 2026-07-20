"""Safetensors weight loading.

Lazy: keeps a name -> shard index and materializes tensors on demand
(needed for models larger than RAM, and for per-expert streaming).

bf16 note: numpy has no bfloat16, so shards are opened with the torch
framework and converted (bf16 -> fp16 keeps the memory footprint; the
mantissa truncation is well below quantization error).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


class WeightStore:
    """Lazy view over all *.safetensors shards in a model directory."""

    def __init__(self, model_path: str | Path, bf16_to: str = "float32"):
        self.model_path = Path(model_path)
        self.bf16_to = bf16_to
        self._index: dict[str, Path] = {}
        self._handles: dict[Path, object] = {}

        index_file = self.model_path / "model.safetensors.index.json"
        if index_file.exists():
            weight_map = json.loads(index_file.read_text())["weight_map"]
            for key, shard in weight_map.items():
                self._index[key] = self.model_path / shard
        else:
            for shard in sorted(self.model_path.glob("*.safetensors")):
                with safe_open(shard, framework="pt") as f:
                    for key in f.keys():
                        self._index[key] = shard
        if not self._index:
            raise FileNotFoundError(f"no safetensors found in {self.model_path}")

    def keys(self):
        return self._index.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def _handle(self, shard: Path):
        if shard not in self._handles:
            self._handles[shard] = safe_open(shard, framework="pt")
        return self._handles[shard]

    def get(self, key: str) -> np.ndarray:
        """Materialize one tensor as a numpy array."""
        import torch

        t = self._handle(self._index[key]).get_tensor(key)
        if t.dtype == torch.bfloat16:
            t = t.to(getattr(torch, self.bf16_to))
        return t.numpy()

    def get_f32(self, key: str) -> np.ndarray:
        t = self.get(key)
        return t if t.dtype == np.float32 else t.astype(np.float32)

    def close(self):
        self._handles.clear()
