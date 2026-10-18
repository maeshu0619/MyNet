"""MPEG pc_error wrapper: GT -> actual final decoded geometry only."""
import re
from pathlib import Path
import numpy as np
from .io import read_vertices, run_command, sha256, write_geometry, _ply_module

NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan)"


def validate_normals(gt, normals):
    xyz, _ = read_vertices(gt)
    normal_xyz, vertices = read_vertices(normals)
    if not np.array_equal(xyz, normal_xyz):
        raise ValueError("Normals PLY coordinates/order must exactly match Original GT")
    if not all(k in vertices.dtype.names for k in ("nx", "ny", "nz")):
        raise ValueError("D2 requires nx/ny/nz; normals are never invented")
    n = np.column_stack([vertices[k] for k in ("nx", "ny", "nz")]).astype(float)
    if not np.isfinite(n).all() or not np.allclose(np.linalg.norm(n, axis=1), 1, atol=1e-3, rtol=0):
        raise ValueError("D2 normals must be finite unit vectors")


def parse_pc_error(output, require_d2=True):
    values = {}
    for tag, suffix in (("d1", "p2point"), ("d2", "p2plane")):
        for name, label in (("psnr", "mseF,PSNR"), ("mse", "mseF")):
            pattern = re.escape(label) + r"\s*\(" + suffix + r"\)\s*:\s*" + NUMBER
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                value = float(matches[-1])
                if np.isnan(value) or value == -np.inf or (name == "mse" and (not np.isfinite(value) or value < 0)):
                    raise ValueError("Invalid pc_error output: " + label)
                values[tag + "_" + name] = value
    for tag in (("d1", "d2") if require_d2 else ("d1",)):
        if tag + "_psnr" not in values or tag + "_mse" not in values:
            raise ValueError("Missing symmetric MPEG {} result; see pc_error stdout".format(tag))
    return values


def evaluate(gt, decoded, folder, binary, resolution, normals=None, require_d2=True, timeout=3600):
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Explicit positive geometry resolution is required")
    if require_d2 and not normals:
        raise ValueError("D2 requested but GT normals are missing")
    folder = Path(folder)
    # pc_error 0.13.4 accepts double xyz but misreads binary double normal fields.
    # Remove any embedded normals from fileA, and supply canonical float32 normals
    # in -n. Neither method gets an independently estimated/decoded normal field.
    metric_gt = folder / "metric_gt.ply"
    xyz, _ = read_vertices(gt)
    write_geometry(metric_gt, xyz)
    metric_normals = None
    if normals:
        validate_normals(gt, normals)
        _, vertices = read_vertices(normals)
        n = np.column_stack([vertices[k] for k in ("nx", "ny", "nz")]).astype(np.float32)
        metric_normals = folder / "metric_normals.ply"
        if not _ply_module().write_ply(str(metric_normals), [xyz, n], ["x", "y", "z", "nx", "ny", "nz"]):
            raise RuntimeError("Canonical normal PLY write failed")
    cmd = [binary, "--fileA=" + str(metric_gt), "--fileB=" + str(decoded),
           "--resolution=" + str(resolution), "--singlePass=0", "--hausdorff=1",
           "--color=0", "--dropdups=2", "--neighborsProc=1", "--averageNormals=1", "--nbThreads=1"]
    if metric_normals:
        cmd.append("--inputNorm=" + str(metric_normals))
    run_command(cmd, folder, "pc_error", timeout)
    output = (Path(folder) / "pc_error.stdout.txt").read_text()
    result = parse_pc_error(output, require_d2)
    warnings = [line.strip() for line in output.splitlines() if "warning" in line.lower()]
    if any("requested neighbors could not be found" in line.lower() for line in warnings):
        raise ValueError("pc_error could not find required neighbors; metric is not accepted")
    result.update(resolution=resolution, metric_binary_sha256=sha256(binary),
                  normals_sha256=sha256(normals) if normals else None,
                  metric_warnings=warnings, canonical_normals_dtype="float32" if normals else None,
                  canonical_normals_sha256=sha256(metric_normals) if metric_normals else None,
                  peak_definition="MPEG pc_error PSNR: 10*log10(3*resolution^2 / symmetric_MSE)",
                  duplicate_policy="pc_error dropdups=2; Original point count retained as bpp denominator",
                  metric_precision="Native tool textual precision; no further rounding")
    return result
