"""Recoverable RAM admission for the opt-in SparsePCGC training route.

No encode is dispatched below the existing safety margin. Only unreferenced
memory is released; the pending input, optimizer and live autograd graph stay
intact. A busy shared host delays the same step instead of inventing a label.
"""
import ctypes
import gc
import time


try:
    _MALLOC_TRIM = ctypes.CDLL('libc.so.6').malloc_trim
    _MALLOC_TRIM.argtypes = [ctypes.c_size_t]
    _MALLOC_TRIM.restype = ctypes.c_int
except (OSError, AttributeError):
    _MALLOC_TRIM = None


def host_recovery_enabled(args):
    return (bool(getattr(args, 'test_route', False))
            and str(getattr(args, 'compress', '')).lower().replace('-', '').replace('_', '') == 'sparsepcgc'
            and bool(getattr(args, 'sparsepcgc_host_pressure_recovery', True)))


def _log(encoder, text):
    writer = getattr(encoder, 'writer', None)
    if writer is not None and hasattr(writer, 'write'):
        writer.write(text)


def reclaim_unused_memory(encoder):
    """Do not evict training/teacher history or start another codec process."""
    gc.collect()
    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)
    result = dict(worker_cleanup='not_running')
    proc = getattr(encoder, '_proc', None)
    if proc is not None and proc.poll() is None and getattr(encoder, '_loaded', False):
        try:
            # We are before the encode request, holding encode_bits' lock.
            # A control RPC must not launch next-frame pool prefetch.
            response, _ = encoder._send_worker_request_once({'command': 'release_runtime_cache'})
            if response.get('status') != 'ok':
                raise RuntimeError(str(response.get('message', response)))
            result = dict(response.get('result', {}), worker_cleanup='ok')
        except Exception as exc:
            # Admission stays closed; this is never an Actual fallback.
            result = dict(worker_cleanup='failed', error=str(exc))
    _log(encoder, 'SparsePCGCHostMemoryReclaim: {}'.format(result))
    return result


def wait_for_host_capacity(encoder, reason):
    minimum = max(int(getattr(encoder.args, 'sparsepcgc_host_min_available_mb', 8192)), 0)
    if minimum <= 0:
        return
    if str(reason) == 'worker_init':
        minimum = max(minimum, 12288)
    available = encoder._host_available_mb()
    if available >= minimum:
        return  # Normal steps have no RPC, GC, delay or extra model work.
    period = max(float(getattr(encoder.args, 'sparsepcgc_host_wait_timeout', 600.)), 0.)
    period = period if period > 0. else 60.
    interval = max(float(getattr(encoder.args, 'sparsepcgc_gpu_wait_interval', 2.)), .1)
    started = time.monotonic()
    next_notice = period
    _log(encoder, 'SparsePCGCHostAdmissionWait: reason={}, available_mb={:.1f}, '
         'required_mb={}, policy=reclaim_and_wait, report_interval_sec={:.1f}'.format(
             reason, available, minimum, period))
    reclaim_unused_memory(encoder)
    while True:
        available = encoder._host_available_mb()
        elapsed = max(time.monotonic() - started, 0.)
        if available >= minimum:
            _log(encoder, 'SparsePCGCHostAdmissionReady: reason={}, waited={:.3f}s, '
                 'available_mb={:.1f}, required_mb={}'.format(reason, elapsed, available, minimum))
            return
        if elapsed >= next_notice:
            _log(encoder, 'SparsePCGCHostAdmissionDeferred: reason={}, waited={:.1f}s, '
                 'available_mb={:.1f}, required_mb={}; preserving the pending step; '
                 'no Actual label or optimizer update'.format(reason, elapsed, available, minimum))
            reclaim_unused_memory(encoder)
            next_notice = elapsed + period
            continue
        # KeyboardInterrupt is intentionally propagated, never swallowed.
        time.sleep(min(interval, max(next_notice - elapsed, .1)))
