"""Persistent GPU expert LRU cache with async upload (pinned staging)."""
from __future__ import annotations

import collections


class GpuExpertCache:
    """Caches expert weight tensors on GPU across layers and requests.

    Keys are arbitrary hashables — use (layer_idx, expert_id, proj_name).
    Values are dicts of CuPy arrays (one per projection) or single arrays.
    """

    def __init__(self, max_bytes: int, device: int = 0):
        import cupy as cp

        self.cp = cp
        self.max_bytes = max_bytes
        self.device = device
        self.used = 0
        self.cache: dict = {}
        self.events: dict = {}  # key -> upload-complete event
        self.lru: collections.OrderedDict = collections.OrderedDict()
        self._graveyard: list = []  # (value, [events]) awaiting last GPU use
        self.hits = 0
        self.misses = 0
        # dedicated copy stream so uploads overlap compute. Blocking
        # (default) streams keep implicit ordering vs the null stream;
        # copy<->compute ordering is handled with events.
        with cp.cuda.Device(device):
            self.copy_stream = cp.cuda.Stream()
        self._pinned_pool = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(self._pinned_pool.malloc)

    def __contains__(self, key) -> bool:
        return key in self.cache

    def _nbytes(self, value) -> int:
        if isinstance(value, dict):
            return sum(v.nbytes for v in value.values())
        return value.nbytes

    def _sweep_graveyard(self):
        still = []
        for value, evs in self._graveyard:
            if all(ev.done for ev in evs):
                continue  # last enqueued use finished; drop -> block returns to pool
            still.append((value, evs))
        self._graveyard = still

    def _evict_until(self, needed: int):
        """Deferred eviction: victims are parked with events recorded on the
        streams that may still touch them, and only released once those
        events complete — no device-wide synchronize on the decode path."""
        cp = self.cp
        self._sweep_graveyard()
        while self.used + needed > self.max_bytes and self.lru:
            victim, _ = self.lru.popitem(last=False)
            value = self.cache.pop(victim)
            self.used -= self._nbytes(value)
            self.events.pop(victim, None)
            evs = []
            for stream in (cp.cuda.get_current_stream(), self.copy_stream):
                ev = cp.cuda.Event()
                ev.record(stream)
                evs.append(ev)
            self._graveyard.append((value, evs))

    def _upload_one(self, arr, stream):
        cp = self.cp
        # allocate with `stream` as the CURRENT stream: CuPy's pool keeps
        # per-stream free lists, so this guarantees the block isn't one a
        # queued kernel on another stream still touches
        with cp.cuda.Device(self.device), stream:
            gpu = cp.empty(arr.shape, dtype=arr.dtype)
            gpu.set(arr, stream=stream)
        return gpu

    def get_or_upload(self, key, cpu_value, stream=None):
        """cpu_value: numpy array or dict of arrays (may be a callable for
        lazy materialization from disk). Returns GPU mirror; upload is
        enqueued on `stream` (defaults to the cache's copy stream)."""
        cp = self.cp
        cur = cp.cuda.get_current_stream()
        if key in self.cache:
            self.lru.move_to_end(key)
            self.hits += 1
            cur.wait_event(self.events[key])  # order after (pre)fetch upload
            return self.cache[key]

        self.misses += 1
        if callable(cpu_value):
            cpu_value = cpu_value()
        stream = stream or self.copy_stream
        needed = self._nbytes(cpu_value)
        self._evict_until(needed)

        if isinstance(cpu_value, dict):
            gpu_value = {k: self._upload_one(v, stream) for k, v in cpu_value.items()}
        else:
            gpu_value = self._upload_one(cpu_value, stream)
        event = cp.cuda.Event()
        event.record(stream)
        cur.wait_event(event)
        self.cache[key] = gpu_value
        self.events[key] = event
        self.lru[key] = None
        self.used += needed
        return gpu_value

    def prefetch(self, key, cpu_value):
        """Non-blocking prefetch on the copy stream (call during attention)."""
        if key not in self.cache:
            self.get_or_upload(key, cpu_value, self.copy_stream)

    def sync(self):
        self.copy_stream.synchronize()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
