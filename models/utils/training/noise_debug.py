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




def prepare_compression_points(clean_xyz, args, model, collect_stats):
    # argsで出力点群ノイズが無効ならcleanな点群をそのまま圧縮損失へ渡す。
    if bool(getattr(args, "disable_output_noise", True)):
        return clean_xyz, empty_noise_debug()
    # 編集後・量子化前だけに一様ノイズを入れる。
    # 形状lossにはclean_xyzを使い、rate/structure lossにはノイズ付き点群を使う。
    return add_uniform_quantization_noise(
        clean_xyz,
        args,
        training=bool(model.training),
        collect_stats=bool(collect_stats),
    )
