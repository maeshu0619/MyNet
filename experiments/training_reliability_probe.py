"""Native codec equality + killed-worker recovery, in an isolated workspace.

Does not train, modify checkpoints, or signal any pre-existing process.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from models.utils.data.dataset import _load_ply_numpy
from models.utils.loss.actual_encoder import _SparsePCGCActualEncoder


class Writer:
    def write(self, value):
        print(value, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("before_actual_encoder", options.before)
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    xyz = torch.from_numpy(_load_ply_numpy(str(options.input), return_color=False)).T.contiguous()
    results = []
    for label, cls in (("before", before._SparsePCGCActualEncoder), ("after", _SparsePCGCActualEncoder)):
        directory = options.out / label
        directory.mkdir()
        args = SimpleNamespace(
            sparsepcgc_root=str(ROOT.parent / "compress/octree/SparsePCGC"),
            sparsepcgc_python="/home/maejima/miniconda3/envs/sparsepcgc/bin/python",
            sparsepcgc_device="cpu", sparsepcgc_tmp_dir=str(directory),
            sparsepcgc_timeout=600, sparsepcgc_mode="dense_lossy",
            sparsepcgc_dense_scale_ae_list=[0], sparsepcgc_dense_scale_sr_list=[2],
            sparsepcgc_scale_ae=0, sparsepcgc_scale_sr=2, sparsepcgc_scale_m=8,
            sparsepcgc_actual_result_cache=False, sparsepcgc_skip_decode=True,
            sparsepcgc_omp_threads=2, sparsepcgc_worker_restart_count=1,
        )
        encoder = cls(args, Writer())
        # A frozen source copy lives outside the project; only relocate its
        # worker script lookup, keeping the real unchanged worker/codec shared.
        encoder._repo_root = lambda: str(ROOT.parent)
        try:
            for attempt in range(2 if label == "after" else 1):
                if attempt:
                    # Kill ONLY the child just launched by this experiment.
                    pid = encoder._proc.pid
                    encoder._proc.kill()
                    encoder._proc.wait(timeout=10)
                    print("[FAULT] own worker killed pid={}".format(pid), flush=True)
                result = encoder.encode_bits(xyz)
                streams = {str(path.relative_to(Path(encoder._workspace_dir) / "encoded")):
                           hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (Path(encoder._workspace_dir) / "encoded").rglob("*")
                           if path.is_file()}
                record = dict(label=label, attempt=attempt, bit=result["bit"],
                              point_count=result["point_count"], streams=streams,
                              launches=encoder._worker_launch_count)
                results.append(record)
                print(json.dumps(record), flush=True)
                (options.out / "measurements.json").write_text(json.dumps(results, indent=2))
        finally:
            encoder.close()
    assert results[0]["bit"] == results[1]["bit"] == results[2]["bit"]
    assert results[0]["streams"] == results[1]["streams"] == results[2]["streams"]
    assert results[2]["launches"] == 2
    (options.out / "success.json").write_text(json.dumps(dict(
        identical_bits=True, identical_stream_hashes=True,
        native_worker_recovery=True, input=str(options.input), original_points=xyz.shape[1],
        scope="codec transport validation; not a model-training or convergence experiment"), indent=2))


if __name__ == "__main__":
    main()
