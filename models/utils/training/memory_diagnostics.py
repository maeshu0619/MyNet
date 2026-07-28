"""訓練中のOS OOM原因を、数値計算へ介入せずCSVへ記録する。"""

from __future__ import annotations

import csv
import gc
import os
import resource
import time
from pathlib import Path

import torch


_COLUMNS = (
    "unix_time", "phase", "episode", "epoch", "step", "global_step", "sample",
    "rss_mb", "rss_anon_mb", "rss_file_mb", "rss_shmem_mb", "vmsize_mb",
    "vmdata_mb", "vmswap_mb", "rss_peak_mb", "threads",
    "children_count", "children_rss_mb", "children_anon_mb",
    "system_mem_available_mb", "system_swap_free_mb",
    "cuda_allocated_mb", "cuda_reserved_mb", "cuda_peak_allocated_mb",
    "cuda_peak_reserved_mb", "gc_gen0", "gc_gen1", "gc_gen2",
    "episode_cache_entries", "episode_cache_mb", "model_cache_entries",
    "model_cache_mb", "ply_cache_entries", "ply_cache_mb",
    "guidance_cpu_entries", "guidance_cpu_mb", "guidance_gpu_entries",
    "den6_payload_entries", "den6_fixed_entries", "actual_gt_cache_entries",
    "surrogate_target_cache_entries", "surrogate_replay_entries",
    "offload_outstanding_count", "offload_outstanding_mb",
    "offload_peak_count", "offload_peak_mb",
    "worker_pid", "worker_rss_mb", "worker_anon_mb",
)


def _kb_to_mb(value):
    return round(float(value or 0) / 1024.0, 3)


def _read_kb_file(path):
    values = {}
    try:
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                parts = line.split()
                if len(parts) >= 2 and parts[1].lstrip("-").isdigit():
                    values[parts[0].rstrip(":")] = int(parts[1])
    except (OSError, ValueError):
        pass
    return values


def _proc_memory(pid):
    status = _read_kb_file(f"/proc/{int(pid)}/status")
    return {
        "rss": status.get("VmRSS", 0),
        "anon": status.get("RssAnon", 0),
        "file": status.get("RssFile", 0),
        "shmem": status.get("RssShmem", 0),
        "vmsize": status.get("VmSize", 0),
        "vmdata": status.get("VmData", 0),
        "vmswap": status.get("VmSwap", 0),
        "threads": status.get("Threads", 0),
    }


def _children(pid):
    try:
        text = Path(f"/proc/{int(pid)}/task/{int(pid)}/children").read_text(
            encoding="utf-8"
        )
    except OSError:
        return []
    result = []
    for item in text.split():
        try:
            child = int(item)
        except ValueError:
            continue
        result.append(child)
        result.extend(_children(child))
    return result


def _safe_len(value):
    try:
        return len(value)
    except (TypeError, AttributeError):
        return 0


class MemoryDiagnosticsCSV:
    """OOM kill直前にも残るよう、各行を即時flushする軽量CSV logger。"""

    def __init__(self, path):
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        self._stream = open(self.path, "a", newline="", encoding="utf-8", buffering=1)
        self._writer = csv.DictWriter(self._stream, fieldnames=_COLUMNS)
        if not exists:
            self._writer.writeheader()
            self._stream.flush()

    def close(self):
        try:
            self._stream.flush()
            self._stream.close()
        except OSError:
            pass

    def record(
        self, phase, *, args=None, model=None, loss=None,
        episode=-1, epoch=-1, step=-1, global_step=-1, sample="",
    ):
        own = _proc_memory(os.getpid())
        child_pids = _children(os.getpid())
        child_stats = [(pid, _proc_memory(pid)) for pid in child_pids]

        worker_pid = -1
        encoder = getattr(loss, "actual_encoder", None) if loss is not None else None
        proc = getattr(encoder, "_proc", None)
        if proc is not None and getattr(proc, "poll", lambda: 1)() is None:
            worker_pid = int(getattr(proc, "pid", -1))
        worker = _proc_memory(worker_pid) if worker_pid > 0 else {}

        model_base = model.module if hasattr(model, "module") else model
        try:
            model_cache = model_base.input_cache_stats()
        except (AttributeError, TypeError):
            model_cache = {}

        try:
            from models.utils.data import dataset as dataset_module
            ply_entries = len(dataset_module._PLY_CACHE)
            ply_bytes = int(dataset_module._PLY_CACHE_BYTES)
        except (ImportError, AttributeError):
            ply_entries = ply_bytes = 0
        try:
            from models.modules import heuristic_guidance
            guidance_cpu_entries = len(heuristic_guidance._EXACT_GUIDANCE_CPU_CACHE)
            guidance_cpu_bytes = int(heuristic_guidance._EXACT_GUIDANCE_CPU_CACHE_BYTES)
            guidance_gpu_entries = len(heuristic_guidance._EXACT_GUIDANCE_CACHE)
        except (ImportError, AttributeError):
            guidance_cpu_entries = guidance_cpu_bytes = guidance_gpu_entries = 0
        try:
            from models.utils.pointcloud import ana_den6_online
            den6_payload_entries = len(ana_den6_online._GLOBAL_PAYLOAD_CACHE)
            den6_fixed_entries = len(ana_den6_online._FIXED_FEATURE_CACHE)
        except (ImportError, AttributeError):
            den6_payload_entries = den6_fixed_entries = 0
        try:
            from models.utils.training.saved_tensor_offload import (
                saved_tensor_offload_stats,
            )
            offload = saved_tensor_offload_stats()
        except (ImportError, AttributeError):
            offload = {}

        meminfo = _read_kb_file("/proc/meminfo")
        gc_count = gc.get_count()
        cuda = torch.cuda.is_available()
        row = {
            "unix_time": f"{time.time():.6f}",
            "phase": str(phase),
            "episode": int(episode), "epoch": int(epoch), "step": int(step),
            "global_step": int(global_step), "sample": str(sample or ""),
            "rss_mb": _kb_to_mb(own["rss"]),
            "rss_anon_mb": _kb_to_mb(own["anon"]),
            "rss_file_mb": _kb_to_mb(own["file"]),
            "rss_shmem_mb": _kb_to_mb(own["shmem"]),
            "vmsize_mb": _kb_to_mb(own["vmsize"]),
            "vmdata_mb": _kb_to_mb(own["vmdata"]),
            "vmswap_mb": _kb_to_mb(own["vmswap"]),
            "rss_peak_mb": _kb_to_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "threads": int(own["threads"]),
            "children_count": len(child_stats),
            "children_rss_mb": _kb_to_mb(sum(item["rss"] for _, item in child_stats)),
            "children_anon_mb": _kb_to_mb(sum(item["anon"] for _, item in child_stats)),
            "system_mem_available_mb": _kb_to_mb(meminfo.get("MemAvailable", 0)),
            "system_swap_free_mb": _kb_to_mb(meminfo.get("SwapFree", 0)),
            "cuda_allocated_mb": round(torch.cuda.memory_allocated() / 1048576.0, 3) if cuda else 0,
            "cuda_reserved_mb": round(torch.cuda.memory_reserved() / 1048576.0, 3) if cuda else 0,
            "cuda_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1048576.0, 3) if cuda else 0,
            "cuda_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1048576.0, 3) if cuda else 0,
            "gc_gen0": gc_count[0], "gc_gen1": gc_count[1], "gc_gen2": gc_count[2],
            "episode_cache_entries": _safe_len(getattr(args, "_episode_input_common_cache", None)),
            "episode_cache_mb": round(float(getattr(args, "_episode_input_common_cache_bytes", 0) or 0) / 1048576.0, 3),
            "model_cache_entries": int(model_cache.get("entries", 0) or 0),
            "model_cache_mb": round(float(model_cache.get("bytes", 0) or 0) / 1048576.0, 3),
            "ply_cache_entries": ply_entries,
            "ply_cache_mb": round(ply_bytes / 1048576.0, 3),
            "guidance_cpu_entries": guidance_cpu_entries,
            "guidance_cpu_mb": round(guidance_cpu_bytes / 1048576.0, 3),
            "guidance_gpu_entries": guidance_gpu_entries,
            "den6_payload_entries": den6_payload_entries,
            "den6_fixed_entries": den6_fixed_entries,
            "actual_gt_cache_entries": _safe_len(getattr(loss, "actual_gt_cache", None)),
            "surrogate_target_cache_entries": _safe_len(getattr(loss, "surrogate_target_cache", None)),
            "surrogate_replay_entries": _safe_len(getattr(loss, "surrogate_replay", None)),
            "offload_outstanding_count": int(offload.get("outstanding_count", 0)),
            "offload_outstanding_mb": round(float(offload.get("outstanding_bytes", 0)) / 1048576.0, 3),
            "offload_peak_count": int(offload.get("peak_count", 0)),
            "offload_peak_mb": round(float(offload.get("peak_bytes", 0)) / 1048576.0, 3),
            "worker_pid": worker_pid,
            "worker_rss_mb": _kb_to_mb(worker.get("rss", 0)),
            "worker_anon_mb": _kb_to_mb(worker.get("anon", 0)),
        }
        self._writer.writerow(row)
        self._stream.flush()
        return row
