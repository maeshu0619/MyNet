"""Bounded RAM storage for immutable tensors, with lossless byte compression."""
import pickle
import zlib

import numpy as np
import torch


class LosslessTensorRAM:
    def __init__(self, max_bytes, max_entries):
        self.max_bytes, self.max_entries = int(max_bytes), int(max_entries)
        self.entries, self.bytes = {}, 0
        self.hits = self.misses = self.rejected = 0

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def put(self, key, values):
        if key in self.entries:
            return True
        if len(self.entries) >= self.max_entries or self.max_bytes <= 0:
            self.rejected += 1
            return False
        encoded, size = {}, 0
        for name, value in values.items():
            if torch.is_tensor(value):
                if value.requires_grad:
                    raise ValueError('LosslessTensorRAM accepts only immutable detached tensors')
                tensor = value.detach().cpu().contiguous()
                payload = zlib.compress(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(), 1)
                encoded[name] = (tensor.dtype, tuple(tensor.shape), payload)
                size += len(payload)
            else:
                encoded[name] = value
        if self.bytes+size > self.max_bytes:
            # Retain the admitted working-set prefix instead of thrashing on
            # cyclic datasets larger than the bounded cache.
            self.rejected += 1
            return False
        self.entries[key] = encoded
        self.bytes += size
        return True

    def get(self, key):
        encoded = self.entries.get(key)
        if encoded is None:
            self.misses += 1
            return None
        self.hits += 1
        result = {}
        for name, value in encoded.items():
            if isinstance(value, tuple) and len(value) == 3 and isinstance(value[0], torch.dtype):
                dtype, shape, payload = value
                raw = np.frombuffer(zlib.decompress(payload), dtype=np.uint8).copy()
                result[name] = torch.from_numpy(raw).view(dtype).reshape(shape)
            else:
                result[name] = value
        return result

    def stats(self):
        return dict(entries=len(self.entries), compressed_bytes=self.bytes,
                    hits=self.hits, misses=self.misses, rejected=self.rejected)


class LosslessObjectRAM:
    """Private-process immutable pool data; never loads external pickle files."""
    def __init__(self, max_bytes, max_entries=128):
        self.max_bytes, self.max_entries = int(max_bytes), int(max_entries)
        self.entries, self.bytes = {}, 0
        self.hits = self.misses = self.rejected = 0

    def put(self, key, value):
        if key in self.entries:
            return True
        if len(self.entries) >= self.max_entries or self.max_bytes <= 0:
            self.rejected += 1
            return False
        payload = zlib.compress(pickle.dumps(value, protocol=4), 1)
        if self.bytes+len(payload) > self.max_bytes:
            self.rejected += 1
            return False
        self.entries[key] = payload
        self.bytes += len(payload)
        return True

    def get(self, key):
        payload = self.entries.get(key)
        if payload is None:
            self.misses += 1
            return None
        self.hits += 1
        return pickle.loads(zlib.decompress(payload))

    def stats(self):
        return dict(entries=len(self.entries), compressed_bytes=self.bytes,
                    hits=self.hits, misses=self.misses, rejected=self.rejected)
