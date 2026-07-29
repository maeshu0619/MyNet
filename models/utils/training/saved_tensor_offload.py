from contextlib import nullcontext
import threading
import weakref

import torch


_OFFLOAD_LOCK = threading.Lock()
_OFFLOAD_OUTSTANDING_BYTES = 0
_OFFLOAD_OUTSTANDING_COUNT = 0
_OFFLOAD_PEAK_BYTES = 0
_OFFLOAD_PEAK_COUNT = 0
_OFFLOAD_HOLDERS = weakref.WeakSet()


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


class _PackedOffloadTensor:
    """autograd nodeが残ってもStep終了時にCPU payloadだけ解放できる容器。"""

    __slots__ = (
        "device",
        "cpu_tensor",
        "tensor_bytes",
        "_finalizer",
        "__weakref__",
    )

    def __init__(self, device, cpu_tensor, tensor_bytes):
        self.device = device
        self.cpu_tensor = cpu_tensor
        self.tensor_bytes = int(tensor_bytes)
        self._finalizer = weakref.finalize(
            cpu_tensor,
            _release_offload_bytes,
            self.tensor_bytes,
        )

    def release(self):
        if self.cpu_tensor is None:
            return False
        # finalizerを明示実行して診断counterも同時に減らす。
        self._finalizer()
        self.cpu_tensor = None
        return True


def release_saved_tensor_offload_payloads():
    """現在Stepでbackward済みのCPU offload payloadを一括解放する。"""
    released_count = 0
    released_bytes = 0
    for holder in list(_OFFLOAD_HOLDERS):
        tensor_bytes = int(holder.tensor_bytes)
        if holder.release():
            released_count += 1
            released_bytes += tensor_bytes
    return {
        "released_count": int(released_count),
        "released_bytes": int(released_bytes),
    }


def _contains_autograd_tensor(value, depth=0, seen=None):
    """診断用のlast/debugコンテナに生きた計算グラフがあるか調べる。"""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen or depth > 8:
        return False
    seen.add(identity)
    if torch.is_tensor(value):
        return bool(value.grad_fn is not None)
    if isinstance(value, dict):
        return any(
            _contains_autograd_tensor(item, depth + 1, seen)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(
            _contains_autograd_tensor(item, depth + 1, seen)
            for item in value
        )
    return False


def _empty_transient_value(value):
    if isinstance(value, dict):
        return {}
    if isinstance(value, list):
        return []
    if isinstance(value, tuple):
        return ()
    return None


def release_autograd_transient_references(model=None, loss=None, args=None):
    """backward後のlast/debug参照だけを切り、次Stepへのgraph蓄積を防ぐ。"""
    released = []
    base_model = model.module if hasattr(model, "module") else model
    modules = list(base_model.modules()) if hasattr(base_model, "modules") else []
    for module in modules:
        for name, value in list(getattr(module, "__dict__", {}).items()):
            if not name.startswith(("last_", "_last_", "debug_")):
                continue
            if not _contains_autograd_tensor(value):
                continue
            setattr(module, name, _empty_transient_value(value))
            released.append(f"{type(module).__name__}.{name}")
    for owner, label in ((loss, "loss"), (args, "args")):
        if owner is None:
            continue
        for name, value in list(getattr(owner, "__dict__", {}).items()):
            if not name.startswith(("last_", "_last_", "_current_")):
                continue
            if not _contains_autograd_tensor(value):
                continue
            setattr(owner, name, _empty_transient_value(value))
            released.append(f"{label}.{name}")
    return released


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
        holder = _PackedOffloadTensor(tensor.device, cpu_tensor, tensor_bytes)
        _OFFLOAD_HOLDERS.add(holder)
        return holder

    def unpack_hook(packed):
        if torch.is_tensor(packed):
            return packed
        if not isinstance(packed, _PackedOffloadTensor):
            raise TypeError(
                f"unexpected saved tensor payload: {type(packed).__name__}"
            )
        if packed.cpu_tensor is None:
            raise RuntimeError(
                "saved tensor CPU payload was released before backward completed"
            )
        return packed.cpu_tensor.to(
            device=packed.device,
            non_blocking=use_pin_memory,
        )

    return torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)
