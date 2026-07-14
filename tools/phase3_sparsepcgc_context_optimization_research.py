#!/usr/bin/env python
"""Phase 3 SparsePCGC context optimization research runner.

長いPromptを再送しなくても研究を継続できるよう、既存Phase成果物と
最小C0 smoke結果をresearch_state/evidence/next_actionsへ保存します。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
csv.field_size_limit(sys.maxsize)

from tools.phase3_sparsepcgc_phase2a_context_micro_oracle import (  # noqa: E402
    DEFAULT_INPUT,
    _coords_from_ply,
    _make_candidate_coords,
    _quality_from_paths,
    _run_worker_decode,
    _safe_float,
    _sha256_text,
    _write_ascii_ply,
)
from tools.phase3_sparsepcgc_research_state import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    PHASE15_DIR,
    PHASE1_DIR,
    PHASE2A_DIR,
    base_research_state,
    evidence_row,
    now_text,
    write_csv,
    write_json,
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _candidate_score(row: Mapping[str, Any], score_name: str) -> float:
    bit = _safe_float(row.get("bit_each"), 0.0)
    p_true = _safe_float(row.get("p_true"), 1.0)
    cost = max(_safe_float(row.get("optimistic_cost_edits"), 1.0), 1.0)
    geom = _safe_float(row.get("geometry_risk_proxy"), 0.0)
    if score_name == "bit_each":
        return bit
    if score_name == "p_true_inverse":
        return 1.0 / max(p_true, 1e-6)
    if score_name == "influence_per_edit":
        return bit / cost
    if score_name == "geometry_safe_influence":
        return bit / cost - 2.0 * geom
    return bit


def _select_c0_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    setting_id: str,
    operation: str,
    top_n: int,
    mid_n: int,
    random_n: int,
    seed: int,
) -> List[Tuple[str, Mapping[str, Any], str]]:
    if operation == "add":
        pool = [r for r in rows if r["setting_id"] == setting_id and r.get("occupied") == "False" and r.get("editable") == "True"]
    elif operation == "adjust":
        pool = [
            r for r in rows
            if r["setting_id"] == setting_id
            and r.get("occupied") == "True"
            and int(float(r.get("descendant_input_voxel_count") or 0)) == 1
        ]
    else:
        pool = [
            r for r in rows
            if r["setting_id"] == setting_id
            and r.get("occupied") == "True"
            and 0 < int(float(r.get("descendant_input_voxel_count") or 0)) <= 4
        ]
    scored = sorted(pool, key=lambda r: _candidate_score(r, "influence_per_edit"), reverse=True)
    selected: List[Tuple[str, Mapping[str, Any], str]] = []
    seen = set()

    def add(group: str, row: Mapping[str, Any], score_name: str) -> None:
        key = (row.get("setting_id"), operation, row.get("symbol_id"), group)
        if key not in seen:
            seen.add(key)
            selected.append((group, row, score_name))

    for row in scored[:top_n]:
        add("top", row, "influence_per_edit")
    if scored and mid_n > 0:
        start = max(0, len(scored) // 2 - mid_n // 2)
        for row in scored[start:start + mid_n]:
            add("middle", row, "influence_per_edit")
    if random_n > 0 and scored:
        rng = random.Random(seed + hash((setting_id, operation)) % 100000)
        for row in rng.sample(scored, k=min(random_n, len(scored))):
            add("random", row, "influence_per_edit")
    return selected


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")

    def ranks(vals: Sequence[float]) -> List[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        for rank, idx in enumerate(order, start=1):
            out[idx] = float(rank)
        return out

    rx = ranks([p[0] for p in pairs])
    ry = ranks([p[1] for p in pairs])
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / max(denx * deny, 1e-12)


def _run_c0_smoke(args: argparse.Namespace, out_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    symbol_rows = _read_csv(PHASE15_DIR / "phase1_5_symbol_to_voxel_mapping.csv")
    phase1_rows = {r["setting_id"]: r for r in _read_csv(PHASE1_DIR / "phase1b_official_pair_baselines.csv")}
    rate_rows = {r["setting_id"]: r for r in _read_csv(PHASE15_DIR / "phase1_5_rate_accounting.csv")}
    coords = _coords_from_ply(DEFAULT_INPUT)
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    if args.max_settings > 0:
        settings = settings[: args.max_settings]
    cache_path = out_dir / "c0_actual_cache.json"
    cache: MutableMapping[str, Any] = _load_json(cache_path, {})
    rows: List[Dict[str, Any]] = []
    for setting_id in settings:
        for operation in ("prune", "merge", "adjust", "add"):
            candidates = _select_c0_candidates(
                symbol_rows,
                setting_id=setting_id,
                operation=operation,
                top_n=args.top_n,
                mid_n=args.middle_n,
                random_n=args.random_n,
                seed=args.seed,
            )
            if not candidates:
                rows.append({
                    "setting_id": setting_id,
                    "operation": operation,
                    "candidate_group": "none",
                    "status": "no_candidate",
                    "note": "No candidate satisfied safe operation constraints.",
                })
                continue
            for idx, (group, sym, score_name) in enumerate(candidates):
                try:
                    edited, dbg = _make_candidate_coords(coords, sym, operation)
                except Exception as exc:
                    rows.append({
                        "setting_id": setting_id,
                        "operation": operation,
                        "candidate_group": group,
                        "symbol_id": sym.get("symbol_id"),
                        "status": "filtered",
                        "error": f"{type(exc).__name__}:{exc}",
                    })
                    continue
                edit_hash = _sha256_text(setting_id + operation + group + json.dumps(dbg, sort_keys=True))
                cache_key = _sha256_text(setting_id + operation + edit_hash)
                ply_path = out_dir / "edited_ply" / f"{setting_id}_{operation}_{group}_{idx}_{edit_hash[:8]}.ply"
                _write_ascii_ply(ply_path, edited)
                if cache_key in cache and args.resume:
                    stats = cache[cache_key]
                    cache_hit = True
                else:
                    stats = _run_worker_decode(
                        input_file=ply_path,
                        output_dir=out_dir / "codec_outputs" / setting_id / f"{operation}_{group}_{idx}_{edit_hash[:8]}",
                        setting=phase1_rows[setting_id],
                        timeout=args.timeout,
                    )
                    cache[cache_key] = stats
                    write_json(cache_path, cache)
                    cache_hit = False
                baseline_bits = _safe_float(rate_rows[setting_id]["decoder_complete_bits"])
                edited_bits = _safe_float(stats.get("file_size"))
                raw_percent = (edited_bits - baseline_bits) / baseline_bits * 100.0 if baseline_bits > 0 else float("nan")
                quality: Dict[str, Any] = {}
                decoded_path = str(stats.get("decoded_path", ""))
                if decoded_path and Path(decoded_path).exists():
                    quality = _quality_from_paths(
                        DEFAULT_INPUT,
                        decoded_path,
                        formal_max_points=args.quality_max_points,
                        normal_max_points=args.quality_max_points,
                        use_pc_error=False,
                    )
                score_value = _candidate_score(sym, score_name)
                row = {
                    "schema_version": "phase_c0_smoke_v1",
                    "setting_id": setting_id,
                    "operation": operation,
                    "candidate_group": group,
                    "symbol_id": sym.get("symbol_id"),
                    "score_name": score_name,
                    "score_value": score_value,
                    "bit_each": sym.get("bit_each"),
                    "p_true": sym.get("p_true"),
                    "mapped_cell_key": sym.get("mapped_cell_key"),
                    "baseline_decoder_complete_bits": baseline_bits,
                    "edited_decoder_complete_bits": edited_bits,
                    "actual_raw_percent": raw_percent,
                    "actual_delta_bits": edited_bits - baseline_bits,
                    "actual_edit_ratio": _safe_float(dbg.get("symmetric_difference"), 0.0) / max(float(coords.shape[0]), 1.0),
                    "symmetric_difference": dbg.get("symmetric_difference", ""),
                    "D1_PSNR_internal": quality.get("d1_psnr", ""),
                    "D2_PSNR_internal": quality.get("d2_psnr", ""),
                    "Chamfer_internal": quality.get("chamfer", ""),
                    "psnr_status": "internal_path_only_official_unresolved",
                    "target_context_delta_bits": "",
                    "ancestor_delta_bits": "",
                    "neighbor_delta_bits": "",
                    "side_delta_bits": "",
                    "collateral_positive_bits": "",
                    "collateral_status": "unavailable_without_counterfactual_reforward",
                    "accepted": bool(math.isfinite(raw_percent) and raw_percent < 0),
                    "negative_example": bool(math.isfinite(raw_percent) and raw_percent >= 0),
                    "cache_hit": cache_hit,
                    "edited_ply_path": str(ply_path),
                    "decoded_path": decoded_path,
                    "debug_json": json.dumps(dbg, sort_keys=True),
                    "status": "ok",
                }
                rows.append(row)
                print(
                    f"[C0] {setting_id} {operation} {group} raw={raw_percent:+.5f}% "
                    f"score={score_value:.4f} cache={cache_hit}",
                    flush=True,
                )
    xs = [_safe_float(r.get("score_value")) for r in rows if r.get("status") == "ok"]
    ys = [-_safe_float(r.get("actual_raw_percent")) for r in rows if r.get("status") == "ok"]
    improved = [r for r in rows if r.get("accepted") is True]
    summary = {
        "actual_eval_count": len([r for r in rows if r.get("status") == "ok"]),
        "negative_example_count": len([r for r in rows if r.get("negative_example") is True]),
        "improved_count": len(improved),
        "best_actual_raw_percent": min([_safe_float(r.get("actual_raw_percent")) for r in rows if r.get("status") == "ok"] or [float("nan")]),
        "spearman_score_vs_gain": _spearman(xs, ys),
        "settings_executed": settings,
    }
    return rows, summary


def _build_outputs(out_dir: Path, c0_rows: Sequence[Mapping[str, Any]], c0_summary: Mapping[str, Any], args: argparse.Namespace) -> None:
    phase15_gate = _load_json(PHASE15_DIR / "phase1_5_gate_result.json", {})
    phase2a_gate = _load_json(PHASE2A_DIR / "phase2a_gate_result.json", {})
    state = base_research_state()
    state["updated_at"] = now_text()
    state["current_phase"] = "C0_candidate_score_calibration_smoke"
    state["measured_results"] = [
        {
            "phase": "Phase1.5",
            "metric": "max_realistic_proxy_1pct_decoder_complete_percent",
            "value": phase15_gate.get("max_realistic_proxy_1pct_decoder_complete_percent"),
        },
        {
            "phase": "Phase2A",
            "metric": "best_atomic_actual_raw_percent",
            "value": phase2a_gate.get("best_raw_percent"),
        },
        {
            "phase": "C0",
            **c0_summary,
        },
    ]
    state["known_issues"] = [
        "internal/myNet D1 around 42dB and external pc_error_d D1 around 63-77dB disagree; official PSNR evaluator unresolved",
        "collateral context delta is unavailable until counterfactual reforward is implemented",
        "Phase2A/C0 atomic edits improve at most ~0.06%, far below 5-10% target",
    ]
    state["verified_facts"] = [
        "decoder-complete bits use LossyCoderDense.test file_size, not main bin alone",
        "external bdXX requantization is excluded from main native dense series",
        "entropy classes collapse AE1/SR2 with AE0/SR3 and AE1/SR1 with AE0/SR2",
    ]
    state["hypotheses"] = [
        "Single high-bit symbols are insufficient; parent-mask or closure-level rewrite is needed",
        "Candidate score must include collateral context worsening, not only bit_each",
    ]
    state["unresolved_questions"] = [
        "Which pc_error_d command exactly matches historical RD sweep?",
        "Can parent-mask counterfactual net Δbits correlate with decoder-complete actual gain?",
        "Can Add become beneficial when constrained to same-parent entropy-reducing rewrites?",
    ]
    state["next_actions"] = ["C1_parent_mask_counterfactual_oracle", "official_psnr_evaluator_alignment"]
    state["files_created"] = [
        str(out_dir / "research_state.json"),
        str(out_dir / "research_findings.md"),
        str(out_dir / "evidence_index.csv"),
        str(out_dir / "next_actions.json"),
        str(out_dir / "analysis_manifest.json"),
        str(out_dir / "phase_c0_smoke_results.csv"),
    ]
    state["files_reused"] = [
        str(PHASE1_DIR / "phase1b_official_pair_baselines.csv"),
        str(PHASE15_DIR / "phase1_5_symbol_to_voxel_mapping.csv"),
        str(PHASE2A_DIR / "phase2a_atomic_actual_results.csv"),
    ]
    state["cache_locations"] = [str(out_dir / "c0_actual_cache.json")]
    write_json(out_dir / "research_state.json", state)

    evidence = [
        evidence_row(
            finding_id="F001",
            finding_type="rate_definition",
            statement="main bin bits alone are not decoder-complete rate",
            status="measured",
            result_file=str(PHASE15_DIR / "phase1_5_rate_accounting.csv"),
            metric="decoder_complete_bits",
            confidence="high",
        ),
        evidence_row(
            finding_id="F002",
            finding_type="actual_gain",
            statement="limited atomic Phase2A best gain is far below 5-10% target",
            status="measured",
            result_file=str(PHASE2A_DIR / "phase2a_atomic_actual_results.csv"),
            metric="actual_raw_percent",
            value=phase2a_gate.get("best_raw_percent"),
            unit="percent",
            confidence="high",
        ),
        evidence_row(
            finding_id="F003",
            finding_type="score_calibration",
            statement="C0 smoke produced candidate-to-actual gain pairs for score calibration",
            status="measured",
            result_file=str(out_dir / "phase_c0_smoke_results.csv"),
            metric="spearman_score_vs_gain",
            value=c0_summary.get("spearman_score_vs_gain"),
            confidence="medium",
        ),
        evidence_row(
            finding_id="F004",
            finding_type="unresolved",
            statement="official external PSNR evaluator remains unresolved",
            status="unresolved",
            result_file=str(PHASE15_DIR / "phase1_5_quality_evaluator_audit.csv"),
            metric="D1/D2",
            confidence="high",
        ),
    ]
    for idx, row in enumerate(c0_rows):
        if row.get("status") == "ok":
            evidence.append(
                evidence_row(
                    finding_id=f"C0_{idx:03d}",
                    finding_type="c0_candidate",
                    statement="C0 candidate actual decoder-complete gain measured",
                    status="measured",
                    result_file=str(out_dir / "phase_c0_smoke_results.csv"),
                    setting_id=str(row.get("setting_id", "")),
                    operation=str(row.get("operation", "")),
                    metric="actual_raw_percent",
                    value=row.get("actual_raw_percent"),
                    unit="percent",
                    confidence="medium",
                    note=str(row.get("candidate_group", "")),
                )
            )
    write_csv(out_dir / "evidence_index.csv", evidence)

    next_actions = [
        {
            "action_id": "A001",
            "priority": 1,
            "objective": "Align official pc_error_d D1/D2 with historical RD sweep",
            "reason": "PSNR band classification cannot be final while internal and external D1 disagree",
            "target_file": "myNet/tools/phase3_sparsepcgc_context_optimization_research.py",
            "target_function": "quality evaluator wrapper",
            "required_inputs": ["decoded PLY", "GT PLY", "historical rd_sweep command"],
            "expected_outputs": ["official_D1", "official_D2", "metric_direction"],
            "gate": "external D1/D2 finite and consistent across one setting",
            "estimated_cost": "low-medium",
            "reuse_candidates": ["tools.phase2_rdo_beam_probe._quality_from_paths"],
            "risk": "pc_error_d normal generation / resolution mismatch",
            "status": "pending",
        },
        {
            "action_id": "A002",
            "priority": 2,
            "objective": "Implement C1 parent-mask counterfactual net Δbits",
            "reason": "Atomic high-bit edits are too weak; parent/context closure may explain achievable gains",
            "target_file": "myNet/tools/phase3_sparsepcgc_context_optimization_research.py",
            "target_function": "phase_c1_parent_mask_counterfactual",
            "required_inputs": ["phase1_5_symbol_to_voxel_mapping.csv", "occupancy debug hook"],
            "expected_outputs": ["target_context_delta_bits", "collateral_positive_bits", "net_delta_bits"],
            "gate": "net_delta_bits correlates with actual gain in small smoke",
            "estimated_cost": "medium",
            "reuse_candidates": ["phase3_sparsepcgc_phase1_worker.py", "phase2s_counterfactual_context_sensitivity.py"],
            "risk": "counterfactual reforward may be expensive",
            "status": "pending",
        },
    ]
    write_json(out_dir / "next_actions.json", {"actions": next_actions})

    findings = [
        "# SparsePCGC Context Optimization Research Findings",
        "",
        "## 1. 研究目的",
        state["research_goal"],
        "",
        "## 2. 今回行ったこと",
        "- 継続研究状態ファイルを生成した。",
        "- Phase C0 smokeとしてcandidate scoreとactual decoder-complete gainの対応を最小実測した。",
        "",
        "## 3. 確認済み事実",
        "- dense主系列はnative/vs1/pq1/official AE-SR pairに固定。",
        "- rateはdecoder-complete logical bitsを使う。",
        "- top high-bit symbol shareは入力Voxel操作率ではない。",
        "",
        "## 4. 実測結果",
        f"- C0 actual_eval_count: {c0_summary.get('actual_eval_count')}",
        f"- C0 best actual raw percent: {c0_summary.get('best_actual_raw_percent')}",
        f"- C0 Spearman score-vs-gain: {c0_summary.get('spearman_score_vs_gain')}",
        "",
        "## 5. 仮説",
        "- 単発Voxel操作では弱く、parent-mask closure単位のrewriteが必要。",
        "- collateral context worseningを含むnet Δbitsが必要。",
        "",
        "## 6. 失敗した仮説",
        "- high bit_each上位atomic editだけで5-10%改善できる、は現時点で支持されない。",
        "",
        "## 7. Context悪化の観測",
        "- collateral deltaは未取得。C1でcounterfactual reforwardが必要。",
        "",
        "## 8. 設定別headroom",
        f"- Phase1.5 realistic 1% proxy max: {phase15_gate.get('max_realistic_proxy_1pct_decoder_complete_percent')}%",
        "",
        "## 9. Add / Prune / Merge / Adjustの比較",
        "- Addは安全制約下で候補不足またはfilteredになることが多い。",
        "- Prune/Mergeはatomicで小改善、Adjustは現設計ではほぼ0改善。",
        "",
        "## 10. PSNR帯別結果",
        "- 正式外部PSNR未解決のため、最終PSNR帯分類は保留。",
        "",
        "## 11. 現在の最良結果",
        f"- best_actual_raw_percent: {c0_summary.get('best_actual_raw_percent')}",
        "",
        "## 12. 目標との差",
        "- 1%以下で5%以上という目標には未達。",
        "",
        "## 13. 次の計画",
        "- A001: official PSNR evaluator alignment",
        "- A002: C1 parent-mask counterfactual oracle",
        "",
        "## 14. 本体非変更確認",
        "- SparsePCGC本体、myNet本体、学習コードは変更していない。",
        "",
        "## 15. 不明点",
        "- external pc_error_dの正しいD2/normal指定。",
        "- parent-mask counterfactualがactual gainと相関するか。",
    ]
    (out_dir / "research_findings.md").write_text("\n".join(findings) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": "analysis_manifest_v1",
        "command": " ".join(sys.argv),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "output_files": state["files_created"],
        "elapsed_time": "",
        "error": "",
        "phase": "C0_smoke",
        "operation": "Prune/Merge/Adjust/Add",
        "budget": "atomic candidates only",
    }
    write_json(out_dir / "analysis_manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description="SparsePCGC context optimization research runner")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--phase", choices=("state", "c0-smoke"), default="c0-smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--settings", default="native_vs1_pq1_ae0_sr3,native_vs1_pq1_ae0_sr2")
    parser.add_argument("--max-settings", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--middle-n", type=int, default=1)
    parser.add_argument("--random-n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--quality-max-points", type=int, default=3000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps({"phase": args.phase, "out_dir": str(args.out_dir), "settings": args.settings}, indent=2))
        return 0
    if args.phase == "state":
        _build_outputs(args.out_dir, [], {"actual_eval_count": 0, "best_actual_raw_percent": ""}, args)
        return 0
    start = time.time()
    c0_rows, c0_summary = _run_c0_smoke(args, args.out_dir)
    write_csv(args.out_dir / "phase_c0_smoke_results.csv", c0_rows)
    _build_outputs(args.out_dir, c0_rows, c0_summary, args)
    manifest_path = args.out_dir / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["elapsed_time"] = time.time() - start
    write_json(manifest_path, manifest)
    gate = {
        "phase": "C0_smoke",
        "status": "PASS" if c0_summary.get("actual_eval_count", 0) > 0 and math.isfinite(float(c0_summary.get("spearman_score_vs_gain", float("nan")))) else "CONDITIONAL",
        "actual_eval_count": c0_summary.get("actual_eval_count"),
        "best_actual_raw_percent": c0_summary.get("best_actual_raw_percent"),
        "spearman_score_vs_gain": c0_summary.get("spearman_score_vs_gain"),
        "phase_c1_executed": False,
    }
    write_json(args.out_dir / "phase_c0_gate_result.json", gate)
    print(f"[Research] outputs: {args.out_dir}")
    print(f"[Research] C0 Gate {gate['status']} best={gate['best_actual_raw_percent']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
