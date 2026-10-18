"""Reuse test.py's loader, preprocessing, inference, hardening and saving verbatim."""
import argparse
import datetime
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    folder = Path(request["folder"])
    import numpy as np
    import torch
    from utils.bdrate.io import save_json, sha256
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.set_num_threads(request["threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import test as existing
    from record.write import Writing
    sys.argv = [str(REPO / "test.py")] + request["network_args"]
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    day, stamp = now.strftime("%Y%m%d"), now.strftime("%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainORtest", default="test")
    args = existing.parse_pugan_args(parser, day, stamp)
    args.trainORtest = "test"
    if str(args.compress).lower().replace("-", "") == "sparsepcgc" and args.test_force_single_plan_student:
        args.heuristic_guidance_mode = "single_plan_student"
        args.heuristic_guidance_enabled = False
        args.heuristic_guidance_network_only_inference = True
        args.single_plan_training_stage = "actual_calibration"
    existing._configure_mp_start_method(args)
    writer = Writing(args, day, stamp, "MyNetwork_test_BDrate", flush_every=1, log_root=str(folder / "network_log"))
    try:
        runtime = existing.configure_knn_backend(args, writer=writer)
        existing.network_module.KNN_BACKEND = runtime
        model = existing._load_trained_model(args, writer)
        saved = existing._load_state_payload(args.ckpt)
        current = model.state_dict()
        missing_trainable = [k for k, p in model.named_parameters() if p.requires_grad and k not in saved]
        mismatches = [k for k, v in saved.items() if k not in current or tuple(v.shape) != tuple(current[k].shape)]
        differences = [k for k, v in saved.items() if k in current and tuple(v.shape) == tuple(current[k].shape)
                       and not torch.equal(v.cpu(), current[k].cpu())]
        audit = dict(missing_trainable=missing_trainable, shape_or_unexpected=mismatches, changed_values=differences)
        save_json(folder / "checkpoint_audit.json", audit)
        if missing_trainable or mismatches or differences:
            raise RuntimeError("Checkpoint is not exact for current inference architecture: missing trainable={}, shape/unexpected={}, changed={}".format(
                missing_trainable[:12], mismatches[:12], differences[:12]))
        # Enforce the same trained-student guards as test.py's main.
        if args.heuristic_guidance_mode == "single_plan_student" and args.test_require_trained_single_plan_student:
            for field in ("single_plan_distillation_updates", "single_plan_actual_training_updates"):
                if field == "single_plan_actual_training_updates" and not args.test_require_actual_calibrated_single_plan_student:
                    continue
                if int(getattr(model, field, torch.zeros(())).cpu()) <= 0:
                    raise RuntimeError("Untrained student checkpoint: " + field)
        del saved, current
        # Existing deterministic subsampling hashes the input pathname. Resolve
        # the evaluation symlink before delegating so a fresh run directory does
        # not change the sampled points / cache identity.
        downsample = existing._downsample_input_batch
        inference = existing._run_named_inference_mode
        def canonical_downsample(points, config, cache_key):
            return downsample(points, config, str(Path(cache_key).resolve()))
        def canonical_inference(mode, net, points, config, path, *rest, **kwargs):
            return inference(mode, net, points, config, str(Path(path).resolve()), *rest, **kwargs)
        existing._downsample_input_batch = canonical_downsample
        existing._run_named_inference_mode = canonical_inference
        # test.test disables training exploration and uses model.eval()/inference_mode.
        # Persist effective args AFTER test() because it applies its own overrides.
        existing.test(model, args, writer)
        save_json(folder / "effective_args.json", {k: v if isinstance(v, (str, int, float, bool, list, tuple, type(None))) else repr(v)
                                                  for k, v in vars(args).items() if not k.startswith("_full")})
        save_json(folder / "inference_provenance.json", dict(
            checkpoint=str(Path(args.ckpt).resolve()), checkpoint_sha256=sha256(args.ckpt),
            test_py_sha256=sha256(REPO / "test.py"), exact_checkpoint_shape_and_values=True,
            seed=0, eval=True, training_exploration=False,
            frozen_encoder_checkpoint_sha256=sha256(REPO / "repkpu_model/ckpt-best.pth") if (REPO / "repkpu_model/ckpt-best.pth").is_file() else None,
            note="Existing test.py pipeline; strict checkpoint check added in standalone wrapper"))
    finally:
        writer.close()


if __name__ == "__main__":
    main()
