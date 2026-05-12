from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from .evaluation import EvaluationConfig, HIGHER_IS_BETTER, SHAPE_METRIC_KEYS, evaluate_decoded_geometry


def parse_shape_eval_args(argv: Sequence[str]) -> Tuple[EvaluationConfig, argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--shape-max-points", type=int, default=0)
    parser.add_argument("--shape-normal-max-points", type=int, default=100000)
    parser.add_argument("--shape-emd-points", type=int, default=2048)
    parser.add_argument("--shape-normal-k", type=int, default=16)
    parser.add_argument("--shape-psnr-peak", type=float, default=0.0)
    parser.add_argument("--no-shape-eval", action="store_true", default=False)
    shape_args, remaining = parser.parse_known_args(list(argv))
    config = EvaluationConfig(
        max_points=max(int(shape_args.shape_max_points), 0),
        normal_max_points=max(int(shape_args.shape_normal_max_points), 0),
        emd_points=max(int(shape_args.shape_emd_points), 1),
        normal_k=max(int(shape_args.shape_normal_k), 3),
        psnr_peak=max(float(shape_args.shape_psnr_peak), 0.0),
    )
    return config, shape_args, remaining


def run_codec_arg_parser(parse_args_func: Callable[[], argparse.Namespace], codec_argv: Sequence[str]) -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]] + list(codec_argv)
        return parse_args_func()
    finally:
        sys.argv = old_argv


def first_decoded_path(decoded_value: object) -> Path:
    text = str(decoded_value)
    if not text or "decode skipped" in text.lower():
        raise FileNotFoundError(f"No decoded path available: {decoded_value}")
    first = text.split(",")[0].strip()
    path = Path(first)
    if not path.exists():
        raise FileNotFoundError(f"Decoded PLY does not exist: {path}")
    return path


def percent_change(value: float, reference: float) -> float:
    value = float(value)
    reference = float(reference)
    if reference == 0.0:
        if value == 0.0:
            return 0.0
        return float("inf") if value > 0.0 else float("-inf")
    return (value - reference) / reference * 100.0


def improvement_percent(key: str, method_value: float, gt_value: float) -> float:
    diff = percent_change(method_value, gt_value)
    return diff if key in HIGHER_IS_BETTER else -diff


def compare_shape_metrics(method_metrics: Mapping[str, float], gt_metrics: Mapping[str, float]) -> Dict[str, float]:
    diff: Dict[str, float] = {}
    for key in SHAPE_METRIC_KEYS:
        diff[f"{key}_diff_percent"] = percent_change(method_metrics[key], gt_metrics[key])
        diff[f"{key}_improvement_percent"] = improvement_percent(key, method_metrics[key], gt_metrics[key])
    return diff


def format_float(value: float, digits: int = 6) -> str:
    value = float(value)
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def format_metric(key: str, value: float, average: bool = False) -> str:
    if key in {"reference_point_count", "decoded_point_count"} and not average:
        return str(int(round(float(value))))
    return format_float(float(value), digits=6)


def evaluate_shape_pair(
    reference_gt_path: str | Path,
    method_decoded_path: str | Path,
    gt_decoded_path: str | Path,
    config: EvaluationConfig,
) -> Dict[str, Mapping[str, float]]:
    method_metrics = evaluate_decoded_geometry(reference_gt_path, method_decoded_path, config)
    gt_metrics = evaluate_decoded_geometry(reference_gt_path, gt_decoded_path, config)
    return {
        "method": method_metrics,
        "gt": gt_metrics,
        "diff": compare_shape_metrics(method_metrics, gt_metrics),
    }


def emit_shape_pair_result(
    index: int,
    method_label: str,
    reference_gt_path: str | Path,
    method_decoded_path: str | Path,
    gt_decoded_path: str | Path,
    shape_result: Mapping[str, Mapping[str, float]],
    emit: Callable[[str, object], None],
    emit_table: Callable[[str, Sequence[object], Iterable[Sequence[object]], object], None],
    writer: object,
) -> None:
    emit(f"shape_pair_index: {index}", writer)
    emit_table(
        "shape_file_info",
        ["field", method_label, "GT"],
        [
            ("reference_original_gt", Path(reference_gt_path).resolve(), Path(reference_gt_path).resolve()),
            ("decoded_path", Path(method_decoded_path).resolve(), Path(gt_decoded_path).resolve()),
            (
                "decoded_point_count",
                format_metric("decoded_point_count", shape_result["method"]["decoded_point_count"]),
                format_metric("decoded_point_count", shape_result["gt"]["decoded_point_count"]),
            ),
        ],
        writer,
    )
    rows = []
    for key in SHAPE_METRIC_KEYS:
        rows.append(
            (
                key,
                format_metric(key, shape_result["method"][key]),
                format_metric(key, shape_result["gt"][key]),
                format_float(shape_result["diff"][f"{key}_diff_percent"]),
                format_float(shape_result["diff"][f"{key}_improvement_percent"]),
            )
        )
    emit_table(
        "shape_metric_comparison",
        ["metric", f"{method_label}_decoded_vs_original_GT", "GT_decoded_vs_original_GT", "diff_percent", "improvement_percent"],
        rows,
        writer,
    )
    emit("", writer)


def emit_shape_summary(
    shape_results: Sequence[Mapping[str, object]],
    method_label: str,
    emit: Callable[[str, object], None],
    emit_table: Callable[[str, Sequence[object], Iterable[Sequence[object]], object], None],
    writer: object,
) -> None:
    if not shape_results:
        return
    average_rows = []
    max_rows = []
    min_rows = []
    for key in SHAPE_METRIC_KEYS:
        method_values = [float(item["shape"]["method"][key]) for item in shape_results]
        gt_values = [float(item["shape"]["gt"][key]) for item in shape_results]
        diff_values = [float(item["shape"]["diff"][f"{key}_diff_percent"]) for item in shape_results]
        improvement_values = [float(item["shape"]["diff"][f"{key}_improvement_percent"]) for item in shape_results]
        finite_indices = [idx for idx, value in enumerate(diff_values) if not math.isnan(value)]
        if not finite_indices:
            max_index = min_index = 0
        else:
            max_index = max(finite_indices, key=lambda idx: diff_values[idx])
            min_index = min(finite_indices, key=lambda idx: diff_values[idx])
        average_rows.append(
            (
                key,
                format_metric(key, sum(method_values) / len(method_values), average=True),
                format_metric(key, sum(gt_values) / len(gt_values), average=True),
                format_float(sum(diff_values) / len(diff_values)),
                format_float(sum(improvement_values) / len(improvement_values)),
            )
        )
        max_rows.append(
            (
                key,
                format_float(diff_values[max_index]),
                Path(str(shape_results[max_index]["method_input"])).name,
                Path(str(shape_results[max_index]["reference_gt"])).name,
                format_metric(key, method_values[max_index]),
                format_metric(key, gt_values[max_index]),
            )
        )
        min_rows.append(
            (
                key,
                format_float(diff_values[min_index]),
                Path(str(shape_results[min_index]["method_input"])).name,
                Path(str(shape_results[min_index]["reference_gt"])).name,
                format_metric(key, method_values[min_index]),
                format_metric(key, gt_values[min_index]),
            )
        )

    emit_table(
        "shape_summary_average",
        ["metric", f"{method_label}_average", "GT_average", "diff_percent_average", "improvement_percent_average"],
        average_rows,
        writer,
    )
    emit("", writer)
    emit_table(
        "shape_summary_diff_percent_max",
        ["metric", "diff_percent", "input_name", "gt_name", f"{method_label}_value", "GT_value"],
        max_rows,
        writer,
    )
    emit("", writer)
    emit_table(
        "shape_summary_diff_percent_min",
        ["metric", "diff_percent", "input_name", "gt_name", f"{method_label}_value", "GT_value"],
        min_rows,
        writer,
    )
