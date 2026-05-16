import torch


def resolve_uniform_noise_delta(args):
    """学習用の量子化ノイズ幅を取得する。

    このノイズはデータ拡張ではなく、codecの量子化誤差を連続的に近似するためのもの。
    通常は共通の --noise_delta を使い、0以下が指定された場合だけ codec 固有の量子化幅へフォールバックする。
    """
    delta = float(getattr(args, "noise_delta", 1.0))
    if delta > 0.0:
        return delta

    compress_key = (
        str(getattr(args, "compress", ""))
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )
    if compress_key == "sparsepcgc":
        return max(float(getattr(args, "sparsepcgc_effective_qs", getattr(args, "qs", 1.0))), 1e-12)
    if compress_key in {"gpcc", "gpcctmc3"}:
        return max(float(getattr(args, "gpcc_effective_qs", getattr(args, "qs", 1.0))), 1e-12)
    if compress_key == "draco":
        return max(float(getattr(args, "draco_effective_qs", getattr(args, "qs", 1.0))), 1e-12)
    return max(float(getattr(args, "qs", 1.0)), 1e-12)


def add_uniform_quantization_noise(points, args, training=True, collect_stats=False):
    """編集後・量子化前の点群にだけ一様ノイズを加える。

    形状損失にはノイズなし点群を使い、rate/structure loss だけがこの点群を見る。
    推論・test・actual codec評価では呼び出しても同一tensorを返し、評価結果にランダム性を入れない。
    """
    use_noise = bool(getattr(args, "use_uniform_noise", True))
    if (not training) or (not use_noise) or points is None:
        return points, {
            "enabled": False,
            "applied": False,
            "delta": 0.0,
            "mean_abs": 0.0,
        }

    delta = resolve_uniform_noise_delta(args)
    if delta <= 0.0:
        return points, {
            "enabled": True,
            "applied": False,
            "delta": float(delta),
            "mean_abs": 0.0,
        }

    half = 0.5 * float(delta)
    noise = torch.empty_like(points).uniform_(-half, half)
    noisy = points + noise
    mean_abs = 0.0
    if collect_stats:
        # ログ用の値だけPython floatへ落とし、GPU tensorや計算グラフを保持しない。
        mean_abs = float(noise.detach().abs().mean().cpu()) if noise.numel() > 0 else 0.0
    return noisy, {
        "enabled": True,
        "applied": True,
        "delta": float(delta),
        "mean_abs": mean_abs,
    }
