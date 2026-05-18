import math
import os

import torch

def case_float(value, default=0.0):
    if torch.is_tensor(value):
        try:
            return float(value.detach().cpu())
        except Exception:
            return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

def case_int(value, default=0):
    try:
        return int(round(case_float(value, default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)

def format_duration_seconds(seconds):
    seconds = case_float(seconds, 0.0)
    if not math.isfinite(seconds) or seconds < 0.0:
        seconds = 0.0
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def process_rss_mb():
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            parts = handle.read().strip().split()
        if len(parts) >= 2:
            return float(int(parts[1]) * page_size) / float(1024 ** 2)
    except Exception:
        return None
    return None

def cuda_alloc_mb(use_cuda):
    if use_cuda and torch.cuda.is_available():
        return float(torch.cuda.memory_allocated()) / float(1024 ** 2)
    return None

def format_xyz_triplet(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().flatten().cpu().tolist()
    try:
        values = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if len(values) < 3:
        return None
    return ",".join(f"{values[i]:.6g}" for i in range(3))

def mean_finite(values):
    finite_values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite_values:
        return None
    return sum(finite_values) / float(len(finite_values))

def summarize_octree_level_debug(level_debug, value_key):
    if not level_debug:
        return None
    chunks = []
    for item in level_debug:
        if not isinstance(item, dict):
            continue
        level = item.get("level", None)
        value = item.get(value_key, None)
        if level is None or value is None:
            continue
        try:
            chunks.append(f"d{int(level)}:{float(value):.6g}")
        except (TypeError, ValueError):
            continue
    return ";".join(chunks) if chunks else None
