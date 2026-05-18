# models/utils/training/correlation.py

import math

import torch


def finite_float_or_none(value):
    try:
        if torch.is_tensor(value):
            value = value.detach().cpu()
        scalar = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(scalar):
        return None

    return scalar


def rolling_pearson(pairs):
    if len(pairs) < 2:
        return None

    xs = [float(pair[0]) for pair in pairs]
    ys = [float(pair[1]) for pair in pairs]

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)

    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    denom = (var_x * var_y) ** 0.5
    if denom <= 1e-12:
        return None

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / denom


def push_rolling_correlation(store, key, metric_value, actual_value, max_samples):
    metric = finite_float_or_none(metric_value)
    actual = finite_float_or_none(actual_value)

    if metric is None or actual is None:
        return None, 0

    pairs = store.setdefault(key, [])
    pairs.append((metric, actual))

    if len(pairs) > max_samples:
        del pairs[:-max_samples]

    return rolling_pearson(pairs), len(pairs)


def format_corr(corr, count):
    if count <= 0:
        return "n/a"

    if corr is None:
        return f"n/a(n={count})"

    return f"{corr:.6f}(n={count})"
