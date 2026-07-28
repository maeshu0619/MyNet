from contextlib import nullcontext
import threading
import weakref

import torch


_OFFLOAD_LOCK = threading.Lock()
_OFFLOAD_OUTSTANDING_BYTES = 0
_OFFLOAD_OUTSTANDING_COUNT = 0
_OFFLOAD_PEAK_BYTES = 0
_OFFLOAD_PEAK_COUNT = 0


def _release_offload_bytes(tensor_bytes):
    global _OFFLOAD_OUTSTANDING_BYTES, _OFFLOAD_OUTSTANDING_COUNT
    with _OFFLOAD_LOCK:
        _OFFLOAD_OUTSTANDING_BYTES = max(
            _OFFLOAD_OUTSTANDING_BYTES - int(tensor_bytes), 0
        )
        _OFFLOAD_OUTSTANDING_COUNT = max(_OFFLOAD_OUTSTANDING_COUNT - 1, 0)


def saved_tensor_offload_stats():
    """未解放autograd offload Tensor量をOOM診断へ返す。"""
    with _OFFLOAD_LOCK:
        return {
            "outstanding_bytes": int(_OFFLOAD_OUTSTANDING_BYTES),
            "outstanding_count": int(_OFFLOAD_OUTSTANDING_COUNT),
            "peak_bytes": int(_OFFLOAD_PEAK_BYTES),
            "peak_count": int(_OFFLOAD_PEAK_COUNT),
        }


def selective_saved_tensor_cpu_offload(
    threshold_mb,
    *,
    pin_memory=False,
    enabled=True,
):
    """指定サイズ以上のCUDA saved tensorだけをCPUへ可逆退避する。"""
    threshold_bytes = max(int(float(threshold_mb) * 1024.0 * 1024.0), 0)
    if (
        not bool(enabled)
        or threshold_bytes <= 0
        or not torch.cuda.is_available()
        or not hasattr(torch.autograd.graph, "saved_tensors_hooks")
    ):
        return nullcontext()

    use_pin_memory = bool(pin_memory)

    def pack_hook(tensor):
        global _OFFLOAD_OUTSTANDING_BYTES, _OFFLOAD_OUTSTANDING_COUNT
        global _OFFLOAD_PEAK_BYTES, _OFFLOAD_PEAK_COUNT
        tensor_bytes = int(tensor.numel()) * int(tensor.element_size())
        if not tensor.is_cuda or tensor_bytes < threshold_bytes:
            return tensor
        cpu_tensor = tensor.detach().to(device="cpu", copy=True)
        if use_pin_memory and not cpu_tensor.is_pinned():
            cpu_tensor = cpu_tensor.pin_memory()
        with _OFFLOAD_LOCK:
            _OFFLOAD_OUTSTANDING_BYTES += tensor_bytes
            _OFFLOAD_OUTSTANDING_COUNT += 1
            _OFFLOAD_PEAK_BYTES = max(
                _OFFLOAD_PEAK_BYTES, _OFFLOAD_OUTSTANDING_BYTES
            )
            _OFFLOAD_PEAK_COUNT = max(
                _OFFLOAD_PEAK_COUNT, _OFFLOAD_OUTSTANDING_COUNT
            )
        weakref.finalize(cpu_tensor, _release_offload_bytes, tensor_bytes)
        return (tensor.device, cpu_tensor)

    def unpack_hook(packed):
        if torch.is_tensor(packed):
            return packed
        device, cpu_tensor = packed
        return cpu_tensor.to(device=device, non_blocking=use_pin_memory)

    return torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)
