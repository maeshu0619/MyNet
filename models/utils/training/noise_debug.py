# models/utils/training/noise_debug.py

import math
import torch

from models.utils.pointcloud.quant_noise import add_uniform_quantization_noise


def empty_noise_debug():
    return {
        "enabled": False,
        "applied": False,
        "delta": 0.0,
        "mean_abs": 0.0,
    }


def merge_noise_debug_values(values):
    valid = [value for value in values if isinstance(value, dict)]
    if not valid:
        return empty_noise_debug()

    mean_values = [
        float(value.get("mean_abs", 0.0))
        for value in valid
        if value.get("mean_abs") is not None
    ]

    return {
        "enabled": any(bool(value.get("enabled", False)) for value in valid),
        "applied": any(bool(value.get("applied", False)) for value in valid),
        "delta": max(float(value.get("delta", 0.0)) for value in valid),
        "mean_abs": float(sum(mean_values) / max(len(mean_values), 1)),
    }


def prepare_compression_points(clean_xyz, args, model, collect_stats):
    # 編集後・量子化前だけに一様ノイズを入れる。
    # 形状lossにはclean_xyzを使い、rate/structure lossにはノイズ付き点群を使う。
    return add_uniform_quantization_noise(
        clean_xyz,
        args,
        training=bool(model.training),
        collect_stats=bool(collect_stats),
    )


def accumulate_compression_terms(term_sums, terms, weight):
    for key, value in (terms or {}).items():
        if torch.is_tensor(value):
            scaled = value * float(weight)
            term_sums[key] = scaled if key not in term_sums else term_sums[key] + scaled
        elif isinstance(value, (int, float)):
            scaled = float(value) * float(weight)
            term_sums[key] = scaled if key not in term_sums else term_sums[key] + scaled
        else:
            term_sums[key] = value