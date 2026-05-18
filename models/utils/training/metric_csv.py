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
    SURROGATE_PRETRAIN_COLUMNS
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


def init_metric_csvs(args, plot, writer):
    paths = {
        "compression_step": None,
        "compression_episode": None,
        "operation_step": None,
        "operation_episode": None,
        "checkpoint_episode": None,
        "surrogate_pretrain_step": None,
    }
    os.makedirs(plot.save_dir, exist_ok=True)
    if bool(getattr(args, "save_compression_metric_csv", True)):
        path = os.path.join(plot.save_dir, f"{args.time}_compression_metrics_step.csv")
        init_csv_file(path, COMPRESSION_METRIC_COLUMNS, writer, "CompressionMetricCSV")
        paths["compression_step"] = path
        epi_path = os.path.join(plot.save_dir, f"{args.time}_compression_metrics_epi.csv")
        init_csv_file(epi_path, COMPRESSION_EPISODE_METRIC_COLUMNS, writer, "CompressionEpisodeMetricCSV")
        paths["compression_episode"] = epi_path
    if bool(getattr(args, "save_operation_metric_csv", True)):
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
    if int(getattr(args, "surrogate_pretrain_steps", 0)) > 0:
        path = os.path.join(plot.save_dir, f"{args.time}_surrogate_pretrain_metrics_step.csv")
        init_csv_file(path, SURROGATE_PRETRAIN_COLUMNS, writer, "SurrogatePretrainMetricCSV")
        paths["surrogate_pretrain_step"] = path
    return paths
