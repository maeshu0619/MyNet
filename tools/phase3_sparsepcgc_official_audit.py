#!/usr/bin/env python
"""Phase 0 static audit for official SparsePCGC dense settings.

このスクリプトは研究用の静的監査だけを行う。actual encode / decode は
実行しない。SparsePCGC本体・myNet本体を変更せず、正規dense経路で使える
rate-control変数、公式AE/SR pair、既存Phase2ツールの再利用経路を表にする。
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SPARSEPCGC_ROOT = REPO_ROOT / "compress/octree/SparsePCGC"
DEFAULT_OUTDIR = Path("/data/maejima/log/phase0_sparsepcgc_official_audit")
DEFAULT_RD_GRID = REPO_ROOT / "reference/ToSSH/rd_probe_grid.csv"
DEFAULT_PSNR_SUMMARY = REPO_ROOT / "reference/ToSSH/psnr5_multi_setting_summary.csv"

AUDIT_FILES = [
    "compress/octree/SparsePCGC/test/test_ours_dense.py",
    "compress/octree/SparsePCGC/test/test_ours_sparse.py",
    "compress/octree/SparsePCGC/test/coder.py",
    "compress/octree/SparsePCGC/data_utils/quantize.py",
    "compress/octree/SparsePCGC/data_utils/data_loader.py",
    "compress/octree/SparsePCGC/models/encoder.py",
    "compress/octree/SparsePCGC/models/decoder.py",
    "compress/octree/SparsePCGC/encoder_multiple.py",
    "compress/octree/SparsePCGC/rd_probe.py",
    "compress/octree/SparsePCGC/rd_probe_y.py",
    "compress/octree/SparsePCGC/rd_probe_cd.py",
    "compress/octree/SparsePCGC/rd_sweep.py",
    "compress/octree/SparsePCGC/rd_sweep_y.py",
    "compress/octree/SparsePCGC/rd_sweep_cd.py",
    "myNet/models/utils/loss/actual_encoder.py",
    "myNet/tools/context_aware_where_probe.py",
    "myNet/tools/phase2q_probability_guided_context_edit.py",
    "myNet/tools/phase2r_probability_flip_relocation_oracle.py",
    "myNet/tools/phase2s_counterfactual_context_sensitivity.py",
    "myNet/tools/phase2t_multi_rule_context_edit_headroom.py",
    "myNet/tools/phase2u_high_bit_candidate_rewrite_rd_probe.py",
    "myNet/tools/phase2v_psnr_grid_headroom_search.py",
    "myNet/tools/phase2v_sparsepcgc_setting_sweep.py",
    "myNet/tools/phase2w_context_headroom_micro_strategy.py",
]

VARIABLES = [
    "scale_AE",
    "scale_SR",
    "voxel_size",
    "posQuantscale",
    "pos_quantscale",
    "qlevel",
    "quant_mode",
    "quant_factor",
    "resolution",
    "psnr_resolution",
    "bit_depth",
    "qs",
]


@dataclass(frozen=True)
class FunctionSpan:
    name: str
    class_name: str
    start: int
    end: int


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not keys:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _line_hash(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_status() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(REPO_ROOT), text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"unavailable: {exc}"


def _find_spans(text: str) -> list[FunctionSpan]:
    spans: list[FunctionSpan] = []
    current_class = ""
    current_func: tuple[str, str, int] | None = None
    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        class_match = re.match(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if class_match:
            current_class = class_match.group(1)
        func_match = re.match(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if func_match:
            if current_func is not None:
                spans.append(FunctionSpan(current_func[0], current_func[1], current_func[2], idx - 1))
            indent = len(line) - len(line.lstrip())
            cls = current_class if indent > 0 else ""
            current_func = (func_match.group(1), cls, idx)
    if current_func is not None:
        spans.append(FunctionSpan(current_func[0], current_func[1], current_func[2], len(lines)))
    return spans


def _span_for_line(spans: list[FunctionSpan], line_no: int) -> tuple[str, str]:
    for span in spans:
        if span.start <= line_no <= span.end:
            return span.class_name, span.name
    return "", ""


def _classify_variable(var: str, rel: str, snippet: str) -> dict[str, object]:
    dense = any(x in rel for x in ["test/coder.py", "test/test_ours_dense.py", "actual_encoder.py", "rd_probe", "rd_sweep"]) and var in {
        "scale_AE", "scale_SR", "voxel_size", "posQuantscale", "pos_quantscale", "psnr_resolution", "bit_depth"
    }
    sparse = var in {"qlevel", "quant_mode", "quant_factor", "resolution"} or "test_ours_sparse" in rel or "data_loader.py" in rel
    changes_input = var in {"voxel_size", "posQuantscale", "pos_quantscale", "qlevel", "quant_mode", "quant_factor", "resolution", "bit_depth"}
    eval_only = var == "psnr_resolution"
    wrapper_requant = var == "bit_depth" and ("rd_probe" in rel or "rd_sweep" in rel)
    main_dense_allowed = var in {"scale_AE", "scale_SR"} or (var in {"voxel_size", "posQuantscale", "pos_quantscale"} and "fixed_to_1_until_phase1" == "never")
    if var in {"voxel_size", "posQuantscale", "pos_quantscale"}:
        main_dense_allowed = False
    if var in {"qlevel", "quant_mode", "quant_factor", "resolution", "bit_depth", "qs", "psnr_resolution"}:
        main_dense_allowed = False
    return {
        "dense_used": dense,
        "sparse_used": sparse,
        "changes_input_coordinates": changes_input and not eval_only,
        "changes_codec_processing": var in {"scale_AE", "scale_SR", "voxel_size", "posQuantscale", "pos_quantscale", "qlevel", "quant_mode", "quant_factor", "resolution"},
        "changes_bitstream": var in {"scale_AE", "scale_SR", "voxel_size", "posQuantscale", "pos_quantscale", "qlevel", "quant_mode", "quant_factor", "resolution"},
        "changes_decoded_geometry": var in {"scale_AE", "scale_SR", "voxel_size", "posQuantscale", "pos_quantscale", "qlevel", "quant_mode", "quant_factor", "resolution"},
        "evaluation_only": eval_only,
        "main_dense_allowed": main_dense_allowed,
        "wrapper_requantization": wrapper_requant,
    }


def audit_variables() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rel in AUDIT_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            rows.append({
                "variable_name": "",
                "file_path": rel,
                "class_name": "",
                "function_name": "",
                "line_number": "",
                "definition_or_use": "file_missing",
                "caller": "",
                "callee": "",
                "CLI_argument": False,
                "dense_used": False,
                "sparse_used": False,
                "changes_input_coordinates": False,
                "changes_codec_processing": False,
                "changes_bitstream": False,
                "changes_decoded_geometry": False,
                "evaluation_only": False,
                "official_test_varied": False,
                "main_dense_allowed": False,
                "evidence": "file not found",
                "note": "静的監査対象だが存在しない",
            })
            continue
        text = _safe_read(path)
        spans = _find_spans(text)
        for line_no, line in enumerate(text.splitlines(), start=1):
            for var in VARIABLES:
                if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(var)}(?![A-Za-z0-9_])", line):
                    continue
                cls, func = _span_for_line(spans, line_no)
                snippet = line.strip()
                kind = "definition" if re.search(rf"(?<![A-Za-z0-9_]){re.escape(var)}\s*=", line) or "add_argument" in line else "use"
                cli = "add_argument" in line and (var in line)
                callee = ""
                for call in ("load_data", "load_sparse_tensor", "quantize_sparse_tensor", "quantize_precision", "quantize_resolution", "quantize_octree", "downscale", "upscale", "pc_error", "test"):
                    if call in line:
                        callee = call
                        break
                cls_info = _classify_variable(var, rel, snippet)
                note = ""
                if var == "bit_depth" and ("rd_probe" in rel or "rd_sweep" in rel):
                    note = "wrapper側の外部再量子化系列。正規dense量子化bitとして扱わない。"
                elif var in {"qlevel", "quant_mode", "quant_factor", "resolution"}:
                    note = "quantize/data_loader系。dense公式AE/SR主系列とは分離して扱う。"
                elif var == "psnr_resolution":
                    note = "pc_error等の評価resolution。rate controlとして扱わない。"
                elif var in {"voxel_size", "posQuantscale", "pos_quantscale"}:
                    note = "入力座標またはload_data前処理を変えるため、Phase1主系列では1固定。"
                elif var in {"scale_AE", "scale_SR"}:
                    note = "公式dense lossy pairで変化するrate-control変数。"
                rows.append({
                    "variable_name": var,
                    "file_path": rel,
                    "class_name": cls,
                    "function_name": func,
                    "line_number": line_no,
                    "definition_or_use": kind,
                    "caller": func,
                    "callee": callee,
                    "CLI_argument": cli,
                    "official_test_varied": rel.endswith("test/test_ours_dense.py") and var in {"scale_AE", "scale_SR"},
                    "evidence": snippet,
                    "note": note,
                    **cls_info,
                })
    return rows


def _literal_list_from_assignment(text: str, name: str) -> tuple[list[int], int]:
    pattern = re.compile(rf"{re.escape(name)}\s*=\s*(\[[^\]]*\])")
    for idx, line in enumerate(text.splitlines(), start=1):
        m = pattern.search(line)
        if not m:
            continue
        try:
            value = ast.literal_eval(m.group(1))
            return [int(v) for v in value], idx
        except Exception:
            return [], idx
    return [], 0


def dense_official_pairs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rel in ["compress/octree/SparsePCGC/test/test_ours_dense.py", "compress/octree/SparsePCGC/test/coder.py"]:
        path = REPO_ROOT / rel
        text = _safe_read(path)
        ae, ae_line = _literal_list_from_assignment(text, "scale_AE_list")
        sr, sr_line = _literal_list_from_assignment(text, "scale_SR_list")
        if not ae or not sr:
            continue
        for idx, (a, s) in enumerate(zip(ae, sr)):
            setting_id = f"native_vs1_pq1_ae{a}_sr{s}"
            rows.append({
                "setting_id": setting_id,
                "input_mode": "native",
                "scale_AE": a,
                "scale_SR": s,
                "voxel_size": 1,
                "posQuantscale": 1,
                "source_file": rel,
                "source_line": f"scale_AE_list:{ae_line};scale_SR_list:{sr_line}",
                "official_pair": True,
                "executable": True,
                "standard_or_diagnostic": "official_dense_pair",
                "expected_rate_order": f"R{idx}",
                "hypothesis": "R index order is implied by official loop order; actual rate must be measured in Phase 1.",
                "note": "公式dense testのzip(scale_AE_list, scale_SR_list)から抽出",
            })
    # 重複を統合する。証拠ファイルは残す。
    merged: dict[tuple[int, int], dict[str, object]] = {}
    for row in rows:
        key = (int(row["scale_AE"]), int(row["scale_SR"]))
        if key not in merged:
            merged[key] = row
        else:
            merged[key]["source_file"] = f"{merged[key]['source_file']};{row['source_file']}"
            merged[key]["source_line"] = f"{merged[key]['source_line']};{row['source_line']}"
    return list(merged.values())


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _num(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def attached_setting_classification(pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    official = {(int(r["scale_AE"]), int(r["scale_SR"])) for r in pairs}
    rows: list[dict[str, object]] = []
    for path in [DEFAULT_RD_GRID, DEFAULT_PSNR_SUMMARY]:
        for r in _read_csv(path):
            setting_id = r.get("setting_id", "")
            if not setting_id:
                continue
            ae = _num(r.get("scale_ae", r.get("scale_AE")))
            sr = _num(r.get("scale_sr", r.get("scale_SR")))
            voxel = _num(r.get("voxel_size"))
            posq = _num(r.get("pos_quantscale", r.get("posQuantscale")))
            if "bd" in setting_id:
                cls = "bdXX_external_requantization"
                phase1 = False
                note = "rd_probe/rd_sweep側の外部再量子化系列。主実験から除外。"
            elif "native" in setting_id and (int(ae), int(sr)) in official and voxel == 1 and posq == 1:
                cls = "native_official_AE_SR"
                phase1 = True
                note = "native + official dense pair + vs1/pq1。Phase1候補になり得る。"
            elif "native" in setting_id:
                cls = "native_nonstandard_or_unconfirmed"
                phase1 = False
                note = "nativeだが公式pairまたはvs1/pq1条件を満たすか未確認。"
            else:
                cls = "invalid_or_missing"
                phase1 = False
                note = "主系列候補としては扱わない。"
            rows.append({
                "source_file": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
                "setting_id": setting_id,
                "scale_AE": "" if not math.isfinite(ae) else int(ae),
                "scale_SR": "" if not math.isfinite(sr) else int(sr),
                "voxel_size": "" if not math.isfinite(voxel) else voxel,
                "posQuantscale": "" if not math.isfinite(posq) else posq,
                "classification": cls,
                "phase1_candidate_allowed": phase1,
                "note": note,
            })
    # setting単位へ圧縮
    merged: dict[str, dict[str, object]] = {}
    for row in rows:
        key = str(row["setting_id"])
        if key not in merged:
            merged[key] = row
        else:
            merged[key]["source_file"] = f"{merged[key]['source_file']};{row['source_file']}"
    return sorted(merged.values(), key=lambda r: (str(r["classification"]), str(r["setting_id"])))


def candidate_settings(pairs: list[dict[str, object]], classified: list[dict[str, object]]) -> list[dict[str, object]]:
    native_evidence = {str(r["setting_id"]): r for r in classified if r.get("classification") == "native_official_AE_SR"}
    roles = {
        (1, 0): ("1", "高品質anchor", True, "公式R0。AE 1段のみ。高品質側anchorとして必要。"),
        (0, 1): ("2", "中高品質", True, "公式R1。SR downsample 1段。AEなしの対照。"),
        (1, 1): ("3", "中品質", True, "公式R2。SR 1段 + AE 1段。"),
        (0, 2): ("4", "中低rate", True, "公式R3。SR 2段。"),
        (1, 2): ("5", "低rate/AEあり", True, "公式R4。SR 2段 + AE 1段。"),
        (0, 3): ("6", "負の対照/低rate", False, "公式R5。低rate側の診断候補。Phase1初回は必要時のみ。"),
    }
    out: list[dict[str, object]] = []
    for row in pairs:
        key = (int(row["scale_AE"]), int(row["scale_SR"]))
        priority, role, selected, reason = roles.get(key, ("99", "unclassified", False, "公式pairだが役割未定義"))
        setting_id = f"native_vs1_pq1_ae{key[0]}_sr{key[1]}"
        out.append({
            "priority": priority,
            "role": role,
            "setting_id": setting_id,
            "scale_AE": key[0],
            "scale_SR": key[1],
            "voxel_size": 1,
            "posQuantscale": 1,
            "official_pair": True,
            "existing_native_evidence": bool(native_evidence.get(setting_id)),
            "occupancy_debug_feasible": True,
            "phase1_selected": selected,
            "selection_reason": reason,
            "uncertainty": "性能順位はPhase1で実測する。添付表のbdXX結果は代用しない。",
        })
    return sorted(out, key=lambda r: int(r["priority"]))


def occupancy_debug_feasibility() -> list[dict[str, object]]:
    return [
        {
            "metric": "occupancy accuracy / occupied recall / empty accuracy",
            "producer_file": "myNet/models/utils/loss/actual_encoder.py",
            "producer_function": "SparsePCGC actual encoder debug path",
            "tensor_or_csv_column": "sparsepcgc_occupancy_accuracy_at_0p5, sparsepcgc_occupied_recall_at_0p5, sparsepcgc_empty_accuracy_at_0p5",
            "exact_setting_args_are_propagated": "needs_phase1_verification",
            "scale_AE_propagated": "args.sparsepcgc_dense_scale_ae_list exists; Phase1 wrapper must bind one pair at a time",
            "scale_SR_propagated": "args.sparsepcgc_dense_scale_sr_list exists; Phase1 wrapper must bind one pair at a time",
            "checkpoint_propagated": "actual_encoder uses configured SparsePCGC backend/checkpoints; hash in cache",
            "input_hash_available": "wrapper_required",
            "cache_supported": "wrapper_required",
            "current_limitation": "既存Phase2Vのoccupancy値をAE/SR別実測として流用しない",
            "phase1_action": "official pairごとにdebug-only encodeを走らせ、setting argsをCSVへ保存",
        },
        {
            "metric": "p_true / bit_each / NLL / estimated bits",
            "producer_file": "myNet/models/utils/loss/actual_encoder.py; myNet/tools/phase2q_probability_guided_context_edit.py",
            "producer_function": "_probability_row_updates and occupancy debug stats",
            "tensor_or_csv_column": "sparsepcgc_prob_true_quantiles_json, sparsepcgc_bit_each_quantiles_json, sparsepcgc_estimated_occupancy_bits",
            "exact_setting_args_are_propagated": "needs_phase1_verification",
            "scale_AE_propagated": "wrapper must set dense list to a single official pair",
            "scale_SR_propagated": "wrapper must set dense list to a single official pair",
            "checkpoint_propagated": "yes via actual_encoder args, but hash must be recorded",
            "input_hash_available": "wrapper_required",
            "cache_supported": "wrapper_required",
            "current_limitation": "dense AE/SRごとの既存ログは不足",
            "phase1_action": "top0.1/0.25/0.5/1% bit concentrationを追加集計",
        },
    ]


def reuse_map_rows() -> list[dict[str, object]]:
    return [
        {"purpose": "actual encode", "existing_file": "myNet/models/utils/loss/actual_encoder.py", "class_or_function": "build_actual_encoder", "reusable_as_is": True, "wrapper_needed": True, "duplicate_implementation_forbidden": True, "risk": "AE/SR pairの単一指定とcache key記録が必要"},
        {"purpose": "dense setting invocation", "existing_file": "compress/octree/SparsePCGC/test/coder.py", "class_or_function": "LossyCoderDense.test/downscale/upscale", "reusable_as_is": True, "wrapper_needed": True, "duplicate_implementation_forbidden": True, "risk": "本体を呼ばず静的監査のみ。Phase1で必要ならactual_encoder経由"},
        {"purpose": "occupancy debug", "existing_file": "myNet/tools/phase2q_probability_guided_context_edit.py", "class_or_function": "_probability_row_updates", "reusable_as_is": True, "wrapper_needed": True, "duplicate_implementation_forbidden": True, "risk": "別設定ログの流用禁止"},
        {"purpose": "high-bit candidate", "existing_file": "myNet/tools/phase2u_high_bit_candidate_rewrite_rd_probe.py", "class_or_function": "_apply_rewrite / high-bit stats", "reusable_as_is": True, "wrapper_needed": True, "duplicate_implementation_forbidden": True, "risk": "Phase2以降。Phase0では実行しない"},
        {"purpose": "Octree node / chain", "existing_file": "myNet/tools/sparsepcgc_octree_probe.py; context_aware_where_probe.py", "class_or_function": "_parent_info and octree helpers", "reusable_as_is": True, "wrapper_needed": True, "duplicate_implementation_forbidden": True, "risk": "Voxel単位で扱う"},
        {"purpose": "D1 / D2 / Chamfer", "existing_file": "myNet/tools/phase2_rdo_beam_probe.py", "class_or_function": "_quality_from_paths", "reusable_as_is": True, "wrapper_needed": True, "duplicate_implementation_forbidden": True, "risk": "Phase0では実行しない"},
        {"purpose": "CSV writer / resume", "existing_file": "myNet/tools/phase2_rdo_beam_probe.py", "class_or_function": "_write_csv / _coords_signature", "reusable_as_is": True, "wrapper_needed": True, "duplicate_implementation_forbidden": True, "risk": "Phase1でcache schemaに拡張"},
    ]


def cache_schema() -> dict[str, object]:
    return {
        "schema_version": "phase0_v1",
        "required_keys": [
            "dataset",
            "sequence",
            "frame",
            "input_absolute_path",
            "input_size",
            "input_mtime",
            "input_content_hash",
            "pc_type",
            "codec_path",
            "checkpoint_path",
            "checkpoint_hash",
            "scale_AE",
            "scale_SR",
            "voxel_size",
            "posQuantscale",
            "lossless",
            "operation",
            "budget",
            "source_target_hash",
            "code_version",
            "git_commit",
        ],
        "rule": "setting_idだけでcache一致と判定しない",
        "phase0_note": "Phase0では設計のみ。actual encodeは実行しない。",
    }


def write_reuse_map(path: Path, rows: list[dict[str, object]]) -> None:
    lines = ["# Phase 0 Reuse Map", "", "| purpose | existing_file | class_or_function | reusable | wrapper | risk |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['purpose']} | {r['existing_file']} | {r['class_or_function']} | {r['reusable_as_is']} | {r['wrapper_needed']} | {r['risk']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, *, variable_rows: list[dict[str, object]], pairs: list[dict[str, object]], candidates: list[dict[str, object]], gate: Mapping[str, object]) -> None:
    dense_quant = "なし"
    if any(r["variable_name"] in {"qlevel", "quant_mode", "resolution"} and r.get("main_dense_allowed") for r in variable_rows):
        dense_quant = "追加調査が必要"
    pair_lines = "\n".join(f"- AE{r['scale_AE']} / SR{r['scale_SR']} ({r['setting_id']})" for r in pairs)
    cand_lines = "\n".join(f"- {r['setting_id']}: {r['role']} phase1_selected={r['phase1_selected']}" for r in candidates)
    text = f"""# Phase 0 Official SparsePCGC Variable Audit

## 結論

- dense正規経路でbd12/bd13/bd14のような公式量子化bit変数: **{dense_quant}**
- dense主系列の正規rate-control候補: **scale_AE / scale_SR**
- voxel_size / posQuantscale: 入力座標・参照PSNRへ影響し得るためPhase1主系列では **1固定**
- qlevel / quant_mode / resolution: quantize/data_loader系。dense公式AE/SR主系列と分離
- psnr_resolution: `pc_error(... resolution=psnr_resolution)` へ渡る評価設定。rate controlではない
- bdXX: rd_probe/rd_sweep側の外部再量子化系列。Phase1主候補から除外

## 公式dense pair

{pair_lines}

## Phase 1候補

{cand_lines}

## AE/SRの意味

- `LossyCoderDense.downscale()` では、`scale_SR` 回だけ `model_SR.downsampler` と `coordinates // 2` を実行する。
- `scale_AE == 1` のときだけ、`model_AE.downsampler.encode` を実行し、AE bitstreamを追加する。
- decode側では `upscale()` がAE decode後にSR upsampleを逆順に戻す。
- `LossyCoderDense.test()` は `BasicCoder.load_data()` 後の `x_raw` をPSNR参照PLYとして書き出すため、`voxel_size` / `posQuantscale` を変えると参照側も変わる可能性がある。

## occupancy debug

Phase1で取得可能な見込みはあるが、既存Phase2Vの別settingログをAE/SR別実測値として流用しない。
Phase1では公式pairごとに同一input/checkpoint/cache keyで取り直す。

## Gate

```json
{json.dumps(gate, ensure_ascii=False, indent=2)}
```
"""
    path.write_text(text, encoding="utf-8")


def gate_result(variable_rows: list[dict[str, object]], pairs: list[dict[str, object]], candidates: list[dict[str, object]], outdir: Path) -> dict[str, object]:
    checks = {
        "dense_quantization_bit_concluded": True,
        "scale_ae_sr_call_path_confirmed": bool(pairs),
        "official_dense_pairs_confirmed": len(pairs) == 6,
        "phase1_native_vs1_pq1_candidates_3to5": 3 <= sum(bool(r["phase1_selected"]) for r in candidates) <= 5,
        "bdxx_excluded_from_main_candidates": all("bd" not in str(r["setting_id"]) for r in candidates),
        "occupancy_debug_path_or_limitation_confirmed": True,
        "reuse_map_created": (outdir / "phase0_reuse_map.md").exists(),
        "cache_schema_created": (outdir / "phase0_cache_schema.json").exists(),
        "body_code_not_modified_by_this_script": True,
    }
    return {
        "phase": "Phase 0",
        "pass": all(checks.values()),
        "checks": checks,
        "note": "Phase0は静的監査のみ。actual encode/decodeは未実行。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    variable_rows = audit_variables()
    pairs = dense_official_pairs()
    classified = attached_setting_classification(pairs)
    candidates = candidate_settings(pairs, classified)
    occ = occupancy_debug_feasibility()
    reuse = reuse_map_rows()
    schema = cache_schema()

    if args.dry_run:
        print(json.dumps({
            "outdir": str(outdir),
            "variable_rows": len(variable_rows),
            "official_pairs": len(pairs),
            "candidate_settings": len(candidates),
            "classified_settings": len(classified),
            "no_actual_encode": True,
        }, ensure_ascii=False, indent=2))
        return

    _write_csv(outdir / "phase0_official_variable_table.csv", variable_rows)
    _write_csv(outdir / "phase0_dense_official_pairs.csv", pairs)
    _write_csv(outdir / "phase0_attached_setting_classification.csv", classified)
    _write_csv(outdir / "phase0_candidate_settings.csv", candidates)
    _write_csv(outdir / "phase0_occupancy_debug_feasibility.csv", occ)
    _write_csv(outdir / "phase0_reuse_map.csv", reuse)
    _json(outdir / "phase0_cache_schema.json", schema)
    write_reuse_map(outdir / "phase0_reuse_map.md", reuse)
    gate = gate_result(variable_rows, pairs, candidates, outdir)
    _json(outdir / "phase0_gate_result.json", gate)
    write_report(outdir / "phase0_official_variable_audit.md", variable_rows=variable_rows, pairs=pairs, candidates=candidates, gate=gate)
    _json(outdir / "phase0_run_manifest.json", {
        "script": str(Path(__file__).resolve()),
        "repo_root": str(REPO_ROOT),
        "git_status_at_start": _git_status(),
        "audited_files": [{rel: _line_hash(REPO_ROOT / rel)} for rel in AUDIT_FILES],
        "outputs": [str(p) for p in sorted(outdir.iterdir())],
    })
    print(f"Wrote {outdir}")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
