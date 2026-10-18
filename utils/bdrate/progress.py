"""Explicit stage progress/timing and completion evidence, independent of Excel."""
import sys
import time
from contextlib import contextmanager


@contextmanager
def timed_stage(label, timings, key, emit):
    emit("[START] " + label)
    started = time.perf_counter()
    try:
        yield
    except BaseException as exc:
        emit("[FAILED] {} ({:.3f}s): {}: {}".format(label, time.perf_counter() - started, type(exc).__name__, exc))
        raise
    else:
        emit("[DONE] {} ({:.3f}s)".format(label, time.perf_counter() - started))
    finally:
        timings[key] = time.perf_counter() - started


def completion_status(rows, summary, metadata, requested):
    required = [r for r in summary if r.get("cloud") == "DATASET_MEAN" and
                (not metadata["config"]["d1_only"] or r["metric"] == "d1")]
    reasons = list(metadata.get("preflight_errors", []))
    if metadata.get("fatal_error") and metadata["fatal_error"] not in reasons:
        reasons.append(metadata["fatal_error"])
    for row in rows:
        if row.get("status") != "ok":
            reasons.append("{} point {}: {}".format(row["series"], row["rate_index"], row.get("error", "failed")))
    for row in summary:
        if row.get("cloud") != "DATASET_MEAN" and (not metadata["config"]["d1_only"] or row["metric"] == "d1") and row["status"] != "ok":
            reasons.append("{} {}: {}".format(row["cloud"], row["metric"].upper(), row.get("error", row["status"])))
    if not metadata.get("protected_source_unchanged", True) or not metadata.get("evaluation_source_unchanged", True) or metadata.get("checkpoint_unchanged") is False:
        reasons.append("Source/checkpoint changed during evaluation")
    measured = sum(r.get("status") == "ok" for r in rows)
    if metadata.get("codec_roundtrips_completed", 0) != requested:
        reasons.append("Actual encode/decode completed {}/{}".format(metadata.get("codec_roundtrips_completed", 0), requested))
    if metadata.get("psnr_measurements_completed", 0) != requested:
        reasons.append("PSNR measurements completed {}/{}".format(metadata.get("psnr_measurements_completed", 0), requested))
    expected_inferences = 0 if metadata["mode"] == "IDENTITY_CONTROL_NOT_TRAINED_MINE" else len(metadata["config"]["input"])
    if metadata.get("mine_inferences_completed", 0) != expected_inferences:
        reasons.append("MINE inference completed {}/{}".format(metadata.get("mine_inferences_completed", 0), expected_inferences))
    okay = bool(required) and all(r["status"] == "ok" for r in required) and measured == requested and not reasons
    status = dict(status="success" if okay else "failed", exit_code=0 if okay else 2,
                  experiment_mode=metadata["mode"], requested_rd_measurements=requested,
                  successful_rd_measurements=measured,
                  successful_codec_roundtrips=metadata.get("codec_roundtrips_completed", 0),
                  successful_psnr_measurements=metadata.get("psnr_measurements_completed", 0),
                  successful_mine_inferences=metadata.get("mine_inferences_completed", 0),
                  elapsed_seconds=metadata.get("elapsed_seconds"), reasons=list(dict.fromkeys(reasons)))
    return status


def print_completion(status, summary, folder):
    out = sys.stdout if status["status"] == "success" else sys.stderr
    print("\n[{}] BD-rate evaluation {}".format("SUCCESS" if status["status"] == "success" else "FAILED",
                                                  "completed" if status["status"] == "success" else "NOT completed successfully"), file=out, flush=True)
    print("MINE inference: {}; Actual encode/decode: {}; PSNR measurements: {}; complete RD measurements: {}/{}".format(
        status["successful_mine_inferences"], status["successful_codec_roundtrips"], status["successful_psnr_measurements"],
        status["successful_rd_measurements"], status["requested_rd_measurements"]), file=out, flush=True)
    if status["successful_rd_measurements"] == 0:
        print("NO successful RD measurements. The Excel file is an ERROR REPORT, not a BD-rate result.", file=out, flush=True)
    if status["experiment_mode"] == "IDENTITY_CONTROL_NOT_TRAINED_MINE":
        print("Identity control only: NO trained MINE model inference was performed.", file=out, flush=True)
    for reason in status["reasons"][:12]:
        print("  - " + reason, file=out, flush=True)
    if len(status["reasons"]) > 12:
        print("  Additional errors: see run_status.json and run.log", file=out, flush=True)
    for row in summary:
        if row.get("cloud") == "DATASET_MEAN" and row["status"] == "ok":
            print("{} dataset BD-rate: {:.10g}%".format(row["metric"].upper(), row["bd_rate_percent"]), file=out, flush=True)
    print("{}: {}".format("Excel results" if status["status"] == "success" else "Excel diagnostic report (NOT evaluation success)", folder / "results.xlsx"), file=out, flush=True)
    print("Detailed status: " + str(folder / "run_status.json"), file=out, flush=True)
