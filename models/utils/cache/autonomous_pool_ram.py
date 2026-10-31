"""Lossless restoration of the autonomous route's immutable candidate pools."""
import gc
import pickle
import zlib
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

from .lossless_tensor_ram import LosslessObjectRAM


@contextmanager
def batch_pool_collection():
    """Collect once after bulk allocation instead of repeatedly scanning it.

    Pools contain plain dictionaries, lists and numbers, with no finalizers or
    cycles. Reference counting remains active. Do not change the caller's GC
    policy when it was already disabled, and restore it even if decoding fails.
    The asynchronous use is restricted to the codec wait: the caller must
    join restoration before returning to any model/loss computation.
    """
    enabled = gc.isenabled()
    if enabled:
        gc.disable()
    try:
        yield
    finally:
        if enabled:
            try:
                # Promote this whole immutable graph together. Leaving it in
                # generation 1 causes the following plan's small allocations
                # to repeatedly scan the freshly restored candidate graph.
                gc.collect(1)
            finally:
                gc.enable()


class AutonomousPoolRAM(LosslessObjectRAM):
    def __init__(self, max_bytes, max_entries=128):
        super().__init__(max_bytes, max_entries)
        self._current_key = None
        self._successors = {}
        self._executor = None
        self._pending = None
        self.prefetch_submitted = self.prefetch_hits = 0

    def get(self, key):
        if self._current_key in self.entries:
            # Only admitted sources own successor entries. Rejected/new frame
            # keys must not grow this side table beyond the cache entry cap.
            self._successors[self._current_key] = key
        self._current_key = key
        payload = self.entries.get(key)
        if payload is None:
            self.misses += 1
            return None
        self.hits += 1
        if self._pending is not None and self._pending[0] == key:
            _, future = self._pending
            self._pending = None
            self.prefetch_hits += 1
            return future.result()
        with batch_pool_collection():
            return pickle.loads(zlib.decompress(payload))

    @staticmethod
    def _restore(payload):
        # No CUDA, tensors, RNG, model state or cache entries are used here.
        # The codec wrapper joins this future before leaving its IPC wait,
        # including on failure, so the model never observes disabled GC.
        with batch_pool_collection():
            return pickle.loads(zlib.decompress(payload))

    def prefetch_next(self):
        """At most one extra decoded pool, using the observed frame sequence.

        The next get still computes and verifies its complete content key.
        A changed frame order therefore falls back to ordinary restoration.
        Start only during the codec IPC wait; join before any further forward
        or backward computation. GC is process-global, so this is mandatory.
        """
        key = self._successors.get(self._current_key)
        if key is None or key not in self.entries:
            return None
        if self._pending is not None:
            if self._pending[0] == key or not self._pending[1].done():
                return self._pending[1]
            self._pending = None
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='autonomous-pool')
        self._pending = (key, self._executor.submit(self._restore, self.entries[key]))
        self.prefetch_submitted += 1
        return self._pending[1]

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._pending = None

    def clear(self):
        self.close()
        self.entries.clear()
        self.bytes = 0
        self._current_key = None
        self._successors.clear()

    def stats(self):
        return dict(super().stats(), prefetch_submitted=self.prefetch_submitted,
                    prefetch_hits=self.prefetch_hits)
