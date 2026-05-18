import math

from .metric_columns import (
    COMPRESSION_EPISODE_METRIC_COLUMNS,
    OPERATION_EPISODE_METRIC_COLUMNS,
    )
from .scalar_utils import case_float


def new_operation_episode_sum():
    return {
        "sums": {key: 0.0 for key in OPERATION_EPISODE_METRIC_COLUMNS},
        "counts": {key: 0 for key in OPERATION_EPISODE_METRIC_COLUMNS},
        "row_count": 0,
        "fresh_actual_count": 0,
        "codec": None,
    }

def accumulate_operation_episode(metric_sums, operation_row):
    metric_sums["row_count"] += 1
    if bool(operation_row.get("fresh_actual", False)):
        metric_sums["fresh_actual_count"] += 1
    if metric_sums.get("codec") is None:
        metric_sums["codec"] = operation_row.get("codec")
    for key in OPERATION_EPISODE_METRIC_COLUMNS:
        if key in {"episode", "stage", "codec", "row_count", "fresh_actual_count"}:
            continue
        value = case_float(operation_row.get(key), float("nan"))
        if math.isfinite(value):
            metric_sums["sums"][key] = float(metric_sums["sums"].get(key, 0.0)) + value
            metric_sums["counts"][key] = int(metric_sums["counts"].get(key, 0)) + 1


def finalize_operation_episode_metrics(episode, stage, metric_sums):
    row = {
        "episode": int(episode) + 1,
        "stage": str(stage),
        "codec": metric_sums.get("codec"),
        "row_count": int(metric_sums.get("row_count", 0)),
        "fresh_actual_count": int(metric_sums.get("fresh_actual_count", 0)),
    }
    for key in OPERATION_EPISODE_METRIC_COLUMNS:
        if key in row:
            continue
        count = int(metric_sums["counts"].get(key, 0))
        row[key] = None if count <= 0 else float(metric_sums["sums"].get(key, 0.0)) / float(count)
    return row


def new_compression_episode_sum():
    return {
        "sums": {key: 0.0 for key in COMPRESSION_EPISODE_METRIC_COLUMNS},
        "counts": {key: 0 for key in COMPRESSION_EPISODE_METRIC_COLUMNS},
        "row_count": 0,
        "fresh_actual_count": 0,
        "codec": None,
        "backend": None,
    }


def accumulate_compression_episode(metric_sums, compression_row):
    metric_sums["row_count"] += 1
    if bool(compression_row.get("fresh_actual", False)):
        metric_sums["fresh_actual_count"] += 1
    if metric_sums.get("codec") is None:
        metric_sums["codec"] = compression_row.get("codec")
    if metric_sums.get("backend") is None:
        metric_sums["backend"] = compression_row.get("backend")
    for key in COMPRESSION_EPISODE_METRIC_COLUMNS:
        if key in {"episode", "stage", "codec", "backend", "row_count", "fresh_actual_count"}:
            continue
        value = case_float(compression_row.get(key), float("nan"))
        if math.isfinite(value):
            metric_sums["sums"][key] = float(metric_sums["sums"].get(key, 0.0)) + value
            metric_sums["counts"][key] = int(metric_sums["counts"].get(key, 0)) + 1


def finalize_compression_episode_metrics(episode, stage, metric_sums):
    row = {
        "episode": int(episode) + 1,
        "stage": str(stage),
        "codec": metric_sums.get("codec"),
        "backend": metric_sums.get("backend"),
        "row_count": int(metric_sums.get("row_count", 0)),
        "fresh_actual_count": int(metric_sums.get("fresh_actual_count", 0)),
    }
    for key in COMPRESSION_EPISODE_METRIC_COLUMNS:
        if key in row:
            continue
        count = int(metric_sums["counts"].get(key, 0))
        row[key] = None if count <= 0 else float(metric_sums["sums"].get(key, 0.0)) / float(count)
    return row
