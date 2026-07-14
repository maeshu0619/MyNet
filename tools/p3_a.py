#!/usr/bin/env python
"""Shared research-state schema for Phase 3 SparsePCGC context optimization.

このファイルは分析runnerからimportされる軽量な状態管理モジュールです。
SparsePCGC本体・myNet本体・学習コードには触れません。
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


SCHEMA_VERSION = "phase3_sparsepcgc_context_optimization_v1"
MASTER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("/data/maejima/log/phase3_sparsepcgc_context_optimization")
PHASE1_DIR = Path("/data/maejima/log/phase1_sparsepcgc_baseline_inventory")
PHASE15_DIR = Path("/data/maejima/log/phase1_5_sparsepcgc_context_reachability")
PHASE2A_DIR = Path("/data/maejima/log/phase2a_sparsepcgc_context_micro_oracle")

OFFICIAL_SETTINGS = [
    {"setting_id": "native_vs1_pq1_ae1_sr0", "scale_AE": 1, "scale_SR": 0},
    {"setting_id": "native_vs1_pq1_ae0_sr1", "scale_AE": 0, "scale_SR": 1},
    {"setting_id": "native_vs1_pq1_ae1_sr1", "scale_AE": 1, "scale_SR": 1},
    {"setting_id": "native_vs1_pq1_ae0_sr2", "scale_AE": 0, "scale_SR": 2},
    {"setting_id": "native_vs1_pq1_ae1_sr2", "scale_AE": 1, "scale_SR": 2},
    {"setting_id": "native_vs1_pq1_ae0_sr3", "scale_AE": 0, "scale_SR": 3},
]


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(MASTER_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except Exception:
        return ""


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})
    tmp.replace(path)


def base_research_state() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": now_text(),
        "git_commit": git_commit(),
        "current_phase": "C0_candidate_score_calibration",
        "completed_phases": ["Phase0", "Phase1", "Phase1.5", "Phase2A_atomic_limited"],
        "failed_gates": [],
        "research_goal": (
            "SparsePCGC本体・decoderを変更せず、native dense入力点群へのごく少数Voxel編集で、"
            "幾何品質を維持しながらdecoder-complete bitsを大幅削減する。"
        ),
        "fixed_codec_conditions": {
            "input": "native",
            "pc_type": "dense",
            "voxel_size": 1,
            "posQuantscale": 1,
            "external_bdxx_requantization": False,
            "qlevel_for_dense": False,
            "psnr_resolution_as_rate_control": False,
            "rate_definition": "decoder_complete_bits",
        },
        "official_settings": OFFICIAL_SETTINGS,
        "selected_settings": ["native_vs1_pq1_ae0_sr2", "native_vs1_pq1_ae0_sr3"],
        "rejected_settings": [
            {
                "setting_id": "bdXX_external_requantization",
                "reason": "外部再量子化系列であり、正規native dense主系列から除外",
            }
        ],
        "entropy_classes": {
            "entropy_class_01": ["native_vs1_pq1_ae1_sr2", "native_vs1_pq1_ae0_sr3"],
            "entropy_class_02": ["native_vs1_pq1_ae1_sr1", "native_vs1_pq1_ae0_sr2"],
        },
        "canonical_psnr_evaluator": {
            "status": "unresolved",
            "temporary_evaluator": "myNet path evaluator from Phase1/Phase2A",
            "issue": "external pc_error_d D1 differs from myNet/internal values and D2 is NaN",
        },
        "canonical_rate_definition": "decoder-complete logical bits returned by LossyCoderDense.test file_size",
        "known_issues": [],
        "verified_facts": [],
        "measured_results": [],
        "hypotheses": [],
        "unresolved_questions": [],
        "next_actions": [],
        "files_created": [],
        "files_reused": [],
        "cache_locations": [],
    }


def evidence_row(
    *,
    finding_id: str,
    finding_type: str,
    statement: str,
    status: str,
    result_file: str = "",
    setting_id: str = "",
    operation: str = "",
    metric: str = "",
    value: Any = "",
    unit: str = "",
    confidence: str = "medium",
    note: str = "",
    source_file: str = "",
    source_function: str = "",
    source_line: str = "",
) -> Dict[str, Any]:
    return {
        "finding_id": finding_id,
        "finding_type": finding_type,
        "statement": statement,
        "status": status,
        "source_file": source_file,
        "source_function": source_function,
        "source_line": source_line,
        "result_file": result_file,
        "result_row_key": setting_id,
        "setting_id": setting_id,
        "sequence": "loot",
        "frame": "1000",
        "operation": operation,
        "metric": metric,
        "value": value,
        "unit": unit,
        "confidence": confidence,
        "note": note,
    }

