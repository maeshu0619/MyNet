"""Auditable artifact IO. Geometry is never normalized for metric evaluation."""
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_vertices(path):
    try:
        from plyfile import PlyData
        vertices = PlyData.read(str(path))["vertex"].data
    except ImportError:
        # Codec-only environments may lack plyfile. Our staged codec input is
        # binary PLY, which the existing test.py reader supports exactly.
        vertices = _ply_module().read_ply(str(path))
    xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float64)
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Empty/nonfinite geometry: {}".format(path))
    return xyz, vertices


def write_geometry(path, xyz, storage_dtype="float64"):
    # Same binary PLY writer as test.py; float64 avoids an extra quantizer.
    write_ply = _ply_module().write_ply
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Invalid geometry")
    stored = xyz.astype(storage_dtype)
    if not np.array_equal(stored.astype(np.float64), xyz):
        raise ValueError("Codec input cannot represent coordinates exactly in " + storage_dtype)
    if Path(path).exists():
        raise FileExistsError(path)
    if not write_ply(str(path), [stored], ["x", "y", "z"]):
        raise RuntimeError("PLY write failed: {}".format(path))


def _ply_module():
    # File import avoids SparsePCGC's unrelated top-level 'models' package.
    import importlib.util
    spec = importlib.util.spec_from_file_location("bd_existing_ply", Path(__file__).resolve().parents[2] / "models/utils/io/utils_ply.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # distinguish +inf (zero error) from missing measurements
    return value


def save_json(path, value):
    # Exclusive creation: existing logs/results are never overwritten.
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def run_command(command, folder, name, timeout=3600, cwd=None, env=None):
    """Keep exact argv, complete stdout/stderr and exit code, including on timeout."""
    folder = Path(folder)
    save_json(folder / (name + ".command.json"), list(map(str, command)))
    status = {}
    started = time.perf_counter()
    with (folder / (name + ".stdout.txt")).open("x") as out, (folder / (name + ".stderr.txt")).open("x") as err:
        try:
            with subprocess.Popen(list(map(str, command)), stdout=out, stderr=err, cwd=cwd, env=env) as proc:
                try:
                    while True:
                        remaining = timeout - (time.perf_counter() - started)
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(command, timeout)
                        try:
                            status["returncode"] = proc.wait(timeout=min(30.0, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            if time.perf_counter() - started >= timeout:
                                raise
                            print("[RUNNING] {}: {:.0f}s elapsed; stdout={}, stderr={}".format(
                                name, time.perf_counter() - started, folder / (name + ".stdout.txt"),
                                folder / (name + ".stderr.txt")), flush=True)
                except BaseException:
                    # Only terminate the subprocess created by THIS evaluation;
                    # existing training/codec processes are never inspected/killed.
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait()
                    raise
        except subprocess.TimeoutExpired:
            status["error"] = "timeout after {} seconds".format(timeout)
        finally:
            status["elapsed_seconds"] = time.perf_counter() - started
            save_json(folder / (name + ".status.json"), status)
    if status.get("returncode") != 0:
        # Surface the actual exception (e.g. checkpoint incompatibility) on the
        # terminal, not just a generic worker exit code. Full logs are preserved.
        with (folder / (name + ".stderr.txt")).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 1600))
            tail = handle.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError("{} failed: {}; see {}{}".format(
            name, status, folder, "\nWorker stderr tail:\n" + tail if tail else ""))


def measure_streams(paths):
    if not paths:
        raise ValueError("No physical bitstream files")
    records = []
    for path in paths:
        path = Path(path)
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError("Missing/empty stream: {}".format(path))
        records.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path)))
    if len({r["path"] for r in records}) != len(records):
        raise ValueError("A stream was counted twice")
    return 8 * sum(r["bytes"] for r in records), records
