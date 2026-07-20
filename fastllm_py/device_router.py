"""Device placement: layers -> devices, experts -> devices.

Replicates fastllm's --device / --moe_device / --moe_device_layers flags.

Device strings: "cuda:0", "cuda:1", "cpu", "numa", "disk".
"numa" and "disk" degrade to "cpu" tiers with different weight residency.
"""
from __future__ import annotations

from dataclasses import dataclass, field


GPU_PREFIX = "cuda"


def _normalize(dev: str) -> str:
    return "cuda:0" if dev == "cuda" else dev


@dataclass
class DeviceMap:
    """Ratio-based layer assignment, e.g. {'cuda:0': 3, 'cuda:1': 2}."""

    ratios: dict[str, float]
    num_layers: int = 0
    _assignment: list[str] = field(default_factory=list)

    def assign(self, num_layers: int) -> list[str]:
        self.num_layers = num_layers
        devs = [(_normalize(d), r) for d, r in self.ratios.items() if r > 0]
        total = sum(r for _, r in devs)
        counts = [int(round(num_layers * r / total)) for _, r in devs]
        # fix rounding drift on the largest bucket
        drift = num_layers - sum(counts)
        counts[counts.index(max(counts))] += drift
        self._assignment = []
        for (dev, _), n in zip(devs, counts):
            self._assignment.extend([dev] * n)
        return self._assignment

    def device_for_layer(self, layer_idx: int) -> str:
        return self._assignment[layer_idx]


@dataclass
class MoeDeviceMap:
    """Expert placement ratios, e.g. {'cuda': 1, 'cpu': 8, 'disk': 1}.

    moe_device_layers: only the LAST k MoE layers use this map; earlier
    MoE layers keep all experts on GPU (fastllm --moe_device_layers).
    """

    ratios: dict[str, float]
    moe_device_layers: int = -1  # -1 = all MoE layers

    def expert_device(self, expert_id: int, num_experts: int) -> str:
        """Static striped assignment of expert id -> device tier."""
        devs = [(d, r) for d, r in self.ratios.items() if r > 0]
        total = sum(r for _, r in devs)
        # cumulative ranges over expert index space
        edge = 0.0
        for dev, r in devs:
            edge += num_experts * r / total
            if expert_id < round(edge):
                return _normalize(dev)
        return _normalize(devs[-1][0])

    def applies_to_layer(self, moe_layer_rank: int, num_moe_layers: int) -> bool:
        if self.moe_device_layers < 0:
            return True
        return moe_layer_rank >= num_moe_layers - self.moe_device_layers
