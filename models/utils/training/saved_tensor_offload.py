from contextlib import nullcontext

import torch


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
        tensor_bytes = int(tensor.numel()) * int(tensor.element_size())
        if not tensor.is_cuda or tensor_bytes < threshold_bytes:
            return tensor
        cpu_tensor = tensor.detach().to(device="cpu", copy=True)
        if use_pin_memory and not cpu_tensor.is_pinned():
            cpu_tensor = cpu_tensor.pin_memory()
        return (tensor.device, cpu_tensor)

    def unpack_hook(packed):
        if torch.is_tensor(packed):
            return packed
        device, cpu_tensor = packed
        return cpu_tensor.to(device=device, non_blocking=use_pin_memory)

    return torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)
