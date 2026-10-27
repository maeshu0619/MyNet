"""Deterministic, balanced training manifests; never include held-out frames."""
import hashlib
import os
import re


def _frame_key(path):
    return tuple((1, int(part)) if part.isdigit() else (0, part)
                 for part in re.split(r"(\d+)", str(path)))


def configure_fixed_episode_frames(seq_datasets, args):
    """Select unique frames once, preserving all_files for the validation split.

    Each dataset receives its own budget. Each selected frame is processed
    once per Episode; matched comparison memory can span Episodes.
    Zero explicitly retains the legacy rotating-window diagnostic mode.
    """
    target = int(getattr(args, "train_fixed_episode_frames", 0))
    if target <= 0:
        return list(seq_datasets), None
    limit = int(getattr(args, "train_frames_per_sequence", 100))
    populations = []
    groups = {}
    seen = set()
    for seq_dir, dataset in seq_datasets:
        files = list(getattr(dataset, "all_files", dataset.files))
        # Apply the established split BEFORE sorting; no held-out frame moves
        # across the existing training/validation boundary.
        if limit > 0:
            files = files[:limit]
        files = sorted(files, key=_frame_key)
        for path in files:
            identity = os.path.realpath(str(path))
            if identity in seen:
                raise ValueError("Duplicate training frame: " + identity)
            seen.add(identity)
        populations.append(files)
        dataset_name = str(getattr(dataset, "training_dataset_name",
                                   getattr(args, "dataname", "dataset")))
        groups.setdefault(dataset_name, []).append(len(populations) - 1)
    if not groups:
        raise ValueError("No training datasets for fixed Episode")
    quotas = [0] * len(populations)
    dataset_records = []
    for dataset_name, indices in groups.items():
        available = sum(len(populations[index]) for index in indices)
        if available < target:
            raise ValueError(
                "Fixed Episode for {} needs {} unique training frames, only {} available; "
                "will not repeat frames or use validation data. "
                "For a short diagnostic use --train_fixed_episode_frames 0."
                .format(dataset_name, target, available)
            )
        remaining = target
        while remaining:
            for index in indices:
                if remaining and quotas[index] < len(populations[index]):
                    quotas[index] += 1
                    remaining -= 1
        dataset_records.append({"dataset": dataset_name, "frames": target,
                                "steps": target,
                                "sequence_steps": [quotas[index] for index in indices]})
    selected = []
    records = []
    for (seq_dir, dataset), population, quota in zip(seq_datasets, populations, quotas):
        if not quota:
            continue
        dataset.fixed_episode_files = tuple(population[:quota])
        dataset.files = list(dataset.fixed_episode_files)
        dataset.active_window_cycle = 0
        dataset.active_window_slot = 0
        dataset.active_window_count = 1
        dataset.active_window_signature = hashlib.sha256(
            "\n".join(map(str, dataset.files)).encode("utf-8")
        ).hexdigest()[:16]
        selected.append((seq_dir, dataset))
        records.append({"sequence": str(seq_dir), "frames": quota,
                        "signature": dataset.active_window_signature,
                        "files": list(map(str, dataset.files))})
    return selected, {"unique_frames_per_episode": target * len(groups),
                      "frames_per_dataset": target,
                      "datasets": dataset_records,
                      "order": "fixed_sequence_then_frame",
                      "sequences": records}
