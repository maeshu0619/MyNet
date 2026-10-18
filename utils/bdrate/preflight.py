"""Read-only evaluation prerequisite checks; never select substitute research data."""
import os
import shutil
from pathlib import Path


def check_prerequisites(args, checkpoint):
    """Collect ALL missing path errors before any inference/codec work starts."""
    errors = []

    def check_file(label, raw, executable=False):
        raw = str(raw)
        found = shutil.which(raw) if executable else None
        path = Path(found or raw).expanduser()
        if not path.is_file():
            message = "{}: file not found: {}".format(label, raw)
            if "/path/" in raw or "/path/to/" in raw:
                message += " (example placeholder; replace with an existing real file)"
            errors.append(message)
        elif executable and not os.access(str(path), os.X_OK):
            errors.append("{}: not executable: {}".format(label, path))
        elif not os.access(str(path), os.R_OK):
            errors.append("{}: not readable: {}".format(label, path))

    for i, path in enumerate(args.input):
        check_file("Original GT [{}]".format(i + 1), path)
    for i, path in enumerate(args.normals):
        check_file("GT normals [{}]".format(i + 1), path)
    check_file("MPEG pc_error", args.pc_error, executable=True)
    if not args.identity_control:
        check_file("MINE Network checkpoint", checkpoint)
        check_file("Network Python", args.network_python, executable=True)
    if args.codec == "SparsePCGC":
        check_file("SparsePCGC Python", args.sparse_python, executable=True)
        for label, value in (("dense", args.sp_checkpoint), ("AE", args.sp_ae_checkpoint), ("SR", args.sp_sr_checkpoint)):
            check_file("SparsePCGC " + label + " checkpoint", value)
    elif args.codec == "OctAttention":
        check_file("OctAttention Python", args.oa_python, executable=True)
        check_file("OctAttention checkpoint", args.oa_checkpoint)
    elif args.codec == "G-PCC":
        check_file("G-PCC binary", args.gpcc_binary, executable=True)
        cfg = Path(args.gpcc_config).expanduser()
        check_file("G-PCC configuration", cfg / "encoder.cfg" if cfg.is_dir() else cfg)
    elif args.codec == "Draco":
        check_file("Draco encoder", args.draco_encoder, executable=True)
        check_file("Draco decoder", args.draco_decoder, executable=True)
    return errors
