"""Explicit real-codec smoke tests, separate from trained-model efficacy.

Creates only a NEW artifact directory. Synthetic surface normals are analytic,
not substituted for a dataset's missing ground-truth normals.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    import numpy as np
    from plyfile import PlyData, PlyElement
    from utils.bdrate.io import save_json
    from utils.bdrate.codecs import encode_decode
    import test_BDrate
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--codecs", nargs="+", default=["G-PCC", "Draco"])
    p.add_argument("--learned-single-point", action="store_true")
    a = p.parse_args()
    folder = Path(a.output).resolve(); folder.mkdir(parents=True, exist_ok=False)
    x, y = np.meshgrid(np.arange(24)*4, np.arange(24)*4)
    x, y = x.ravel(), y.ravel()
    z = np.rint(.004*x*x + .006*y*y + .002*x*y)
    normals = np.column_stack((-.008*x - .002*y, -.012*y - .002*x, np.ones(len(x))))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    v = np.empty(len(x), dtype=[(k, "f4") for k in ("x", "y", "z", "nx", "ny", "nz")])
    for name, values in zip(v.dtype.names, (x, y, z, *normals.T)):
        v[name] = values
    gt = folder / "synthetic_surface.ply"
    PlyData([PlyElement.describe(v, "vertex")]).write(str(gt))
    results = []
    for codec in a.codecs:
        args = test_BDrate.parse_args(["--input", str(gt), "--resolution", "255", "--codec", codec,
                                      "--identity-control", "--dataset", "SYNTHETIC_SELF_TEST_NOT_THESIS",
                                      "--output", str(folder / codec), "--timeout", "300"])
        if a.learned_single_point:
            dest = folder / codec; dest.mkdir()
            point = {"OctAttention": dict(qs=8), "SparsePCGC": dict(scale_ae=1, scale_sr=1)}[codec]
            try:
                result = encode_decode(codec, gt, dest, point, vars(args))
                from utils.bdrate.metrics import evaluate
                result.update(evaluate(gt, result["decoded_path"], dest, args.pc_error, 255, str(gt)))
                result.update(codec=codec, status="ok")
            except Exception as exc:
                result = dict(codec=codec, status="error", error=repr(exc))
            save_json(dest / "smoke_result.json", result)
            results.append(result)
        else:
            command = [sys.executable, "-B", str(REPO / "test_BDrate.py"), "--input", str(gt), "--resolution", "255",
                       "--codec", codec, "--identity-control", "--dataset", "SYNTHETIC_SELF_TEST_NOT_THESIS",
                       "--output", str(folder / codec), "--timeout", "300"]
            completed = subprocess.run(command, cwd=REPO, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            results.append(dict(codec=codec, exit_code=completed.returncode))
    save_json(folder / "smoke_summary.json", results)
    print(json.dumps([{k: r[k] for k in ("codec", "status", "exit_code", "error") if k in r} for r in results]), flush=True)
    return 0 if all(r.get("status") == "ok" or r.get("exit_code") == 0 for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
