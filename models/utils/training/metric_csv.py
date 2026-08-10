import csv
import math
import os
import torch

from .scalar_utils import *
from .correlation_debug import *
from .sparsepcgc_controls import *
from .compression_primary_loss import *
from .case_debug import *
from .actual_codec_status import *
from .metric_rows import *
from .episode_metrics import *
from .checkpoint_metrics import *
from .metric_columns import (
    CASE_DEBUG_COLUMNS, 
    COMPRESSION_METRIC_COLUMNS, 
    COMPRESSION_EPISODE_METRIC_COLUMNS, 
    OPERATION_METRIC_COLUMNS, 
    OPERATION_EPISODE_METRIC_COLUMNS, 
    CHECKPOINT_METRIC_COLUMNS, 
    CHECKPOINT_AVG_KEYS, 
    SURROGATE_PRETRAIN_COLUMNS,
    LOSS_GRAD_PROBE_COLUMNS,
    PROPOSAL_CANDIDATE_COLUMNS,
    FULL_CLOUD_AMOUNT_CANDIDATE_COLUMNS,
    )

def init_csv_file(path, columns, writer, label):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=columns).writeheader()
    writer.write(f"{label}: enabled path={path}")


def append_csv_row(path, columns, row):
    if not path:
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=columns).writerow(
            {key: row.get(key, None) for key in columns}
        )


def plot_fixed_validation_curve(path):
    """同一フレーム集合のActual/Geometryだけを学習曲線として描画する。"""
    if not path or not os.path.exists(path):
        return None
    episodes = []
    actual = []
    objective = []
    geometry = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                episode = int(row.get("episode", ""))
                actual_value = float(row.get("full_cloud_val_actual_percent", ""))
                objective_value = float(row.get("full_cloud_val_fixed_objective", ""))
                geometry_value = float(row.get("full_cloud_val_geometry", ""))
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (
                actual_value, objective_value, geometry_value
            )):
                continue
            episodes.append(episode)
            actual.append(actual_value)
            objective.append(objective_value)
            geometry.append(geometry_value)
    if not episodes:
        return None
    import matplotlib.pyplot as plt

    output_path = path.replace(
        "_checkpoint_metrics_epi.csv", "_fixed_validation_rd.png"
    )
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(episodes, actual, label="Actual Compression Loss (%)")
    axes[0].plot(episodes, objective, label="Fixed RD Objective")
    axes[0].set_ylabel("Lower is better")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(episodes, geometry, label="Geometry Loss", color="tab:green")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Geometry")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def init_metric_csvs(args, plot, writer):
    paths = {
        "compression_step": None,
        "compression_episode": None,
        "operation_step": None,
        "operation_episode": None,
        "checkpoint_episode": None,
        "surrogate_pretrain_step": None,
        "loss_grad_probe": None,
        "proposal_candidate_step": None,
        "full_cloud_amount_candidate_step": None,
    }
    os.makedirs(plot.save_dir, exist_ok=True)
    save_step_csv = bool(getattr(args, "save_step_metric_csv", False))
    if bool(getattr(args, "save_compression_metric_csv", True)):
        if save_step_csv:
            path = os.path.join(plot.save_dir, f"{args.time}_compression_metrics_step.csv")
            init_csv_file(path, COMPRESSION_METRIC_COLUMNS, writer, "CompressionMetricCSV")
            paths["compression_step"] = path
        epi_path = os.path.join(plot.save_dir, f"{args.time}_compression_metrics_epi.csv")
        init_csv_file(epi_path, COMPRESSION_EPISODE_METRIC_COLUMNS, writer, "CompressionEpisodeMetricCSV")
        paths["compression_episode"] = epi_path
    if bool(getattr(args, "save_operation_metric_csv", True)):
        if save_step_csv:
            path = os.path.join(plot.save_dir, f"{args.time}_operation_metrics_step.csv")
            init_csv_file(path, OPERATION_METRIC_COLUMNS, writer, "OperationMetricCSV")
            paths["operation_step"] = path
        epi_path = os.path.join(plot.save_dir, f"{args.time}_operation_metrics_epi.csv")
        init_csv_file(epi_path, OPERATION_EPISODE_METRIC_COLUMNS, writer, "OperationEpisodeMetricCSV")
        paths["operation_episode"] = epi_path
    if bool(getattr(args, "save_checkpoint_metric_csv", True)):
        path = os.path.join(plot.save_dir, f"{args.time}_checkpoint_metrics_epi.csv")
        init_csv_file(path, CHECKPOINT_METRIC_COLUMNS, writer, "CheckpointMetricCSV")
        paths["checkpoint_episode"] = path
    if save_step_csv and int(getattr(args, "surrogate_pretrain_steps", 0)) > 0:
        path = os.path.join(plot.save_dir, f"{args.time}_surrogate_pretrain_metrics_step.csv")
        init_csv_file(path, SURROGATE_PRETRAIN_COLUMNS, writer, "SurrogatePretrainMetricCSV")
        paths["surrogate_pretrain_step"] = path
    if save_step_csv and bool(getattr(args, "loss_grad_probe_enabled", False)):
        path = os.path.join(plot.save_dir, f"{args.time}_step_grad.csv")
        init_csv_file(path, LOSS_GRAD_PROBE_COLUMNS, writer, "StepGradCSV")
        paths["loss_grad_probe"] = path
    if save_step_csv and bool(getattr(args, "sparsepcgc_algorithmic_proposal_selector", True)):
        path = os.path.join(plot.save_dir, f"{args.time}_proposal_candidate_metrics_step.csv")
        init_csv_file(path, PROPOSAL_CANDIDATE_COLUMNS, writer, "ProposalCandidateMetricCSV")
        paths["proposal_candidate_step"] = path
    if save_step_csv and str(getattr(args, "sparsepcgc_training_mode", "subtree_selector")).strip().lower() == "full_cloud_amount":
        path = os.path.join(plot.save_dir, f"{args.time}_full_cloud_amount_candidate_metrics_step.csv")
        init_csv_file(path, FULL_CLOUD_AMOUNT_CANDIDATE_COLUMNS, writer, "FullCloudAmountCandidateMetricCSV")
        paths["full_cloud_amount_candidate_step"] = path
    return paths
