"""Geometry-only native adapters; training teacher wrappers are NOT rate oracles.

The existing command builders are reused, but their training prequantization,
origin removal, temporary-file deletion and proxy-rate fallbacks are bypassed.
"""
import os
import sys
from pathlib import Path
from .io import measure_streams, read_vertices, run_command, save_json, sha256, write_geometry

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO.parent / "compress" / "octree"


def default_points(codec):
    if codec == "G-PCC":
        return [dict(positionQuantizationScale=v) for v in (.125, .25, .5, .75, .875, .9375)]
    if codec == "Draco":
        return [dict(qp=v) for v in (5, 6, 7, 8, 9, 10)]
    if codec == "OctAttention":
        return [dict(qs=v) for v in (8, 6, 4, 3, 2, 1)]
    if codec == "SparsePCGC":
        return [dict(scale_ae=a, scale_sr=s) for a, s in zip((1, 0, 1, 0, 1, 0), (0, 1, 1, 2, 2, 3))]
    raise ValueError("Unsupported codec: " + codec)


def validate_points(codec, points):
    if len(points) < 4:
        raise ValueError("At least four RD settings are required; six recommended")
    expected = {"G-PCC": {"positionQuantizationScale"}, "Draco": {"qp"},
                "OctAttention": {"qs"}, "SparsePCGC": {"scale_ae", "scale_sr", "voxel_size"}}[codec]
    required = expected - {"voxel_size"}
    import math
    import json
    seen = set()
    for p in points:
        if not isinstance(p, dict) or set(p) - expected or not required.issubset(p):
            raise ValueError("Invalid {} point: {}".format(codec, p))
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in p.values()):
            raise ValueError("RD parameters must be finite numbers")
        key = json.dumps(p, sort_keys=True)
        if key in seen:
            raise ValueError("Duplicate codec setting")
        seen.add(key)
        if codec == "Draco" and (int(p["qp"]) != p["qp"] or not 1 <= p["qp"] <= 30):
            raise ValueError("Draco qp must be an integer in 1..30")
        if codec == "G-PCC" and not 0 < p["positionQuantizationScale"] <= 1:
            raise ValueError("G-PCC sequence quantization scale must be in (0,1]")
        if codec == "OctAttention" and p["qs"] <= 0:
            raise ValueError("OctAttention qs must be positive")
        if codec == "SparsePCGC" and (p["scale_ae"] not in (0, 1) or int(p["scale_sr"]) != p["scale_sr"] or p["scale_sr"] < 0 or p.get("voxel_size", 1) <= 0):
            raise ValueError("Invalid SparsePCGC AE/SR/voxel setting")


def encode_decode(codec, source, folder, point, settings):
    folder = Path(folder)
    xyz, _ = read_vertices(source)
    # Strip attributes/normals without normalizing or dropping points.
    geometry = folder / "geometry_input.ply"
    # Native Draco's PLY loader accepts float32/int32 only. Fail on unrepresentable
    # float64 inputs instead of silently introducing an extra lossy conversion.
    write_geometry(geometry, xyz, "float32" if codec == "Draco" else "float64")
    decoded = folder / "decoded.ply"
    timeout = settings["timeout"]
    streams, metadata = [], {}
    if codec == "G-PCC":
        from models.utils.compression.gpcc_tmc3 import GPCCGeometryEncoder
        encoder = GPCCGeometryEncoder(str(ROOT / "G-PCC"), settings["gpcc_binary"], settings["gpcc_config"])
        encoder.validate()
        bitstream = folder / "geometry.bin"
        cmd = encoder._command(str(geometry), str(bitstream), str(folder / "encoder_reconstruction.ply"))
        # Native quantization retains origin and serializes the required coordinate system.
        cmd = [x for x in cmd if not x.startswith("--positionQuantizationScale=")]
        cmd += ["--positionQuantizationScale=" + str(point["positionQuantizationScale"]),
                "--outputBinaryPly=1", "--outputScaling=1"]
        run_command(cmd, folder, "encode", timeout, cwd=encoder.root)
        run_command([encoder.encoder_path, "--mode=1", "--compressedStreamPath=" + str(bitstream),
                     "--reconstructedDataPath=" + str(decoded), "--outputBinaryPly=1", "--outputScaling=1"],
                    folder, "decode", timeout, cwd=encoder.root)
        streams = [bitstream]
        metadata.update(binary_sha256=sha256(encoder.encoder_path), config_sha256=sha256(encoder.cfg_path),
                        rate_definition="complete geometry-only native stream including headers")
    elif codec == "Draco":
        from models.utils.compression.draco import DracoGeometryEncoder
        encoder = DracoGeometryEncoder(str(ROOT / "Draco"), settings["draco_encoder"], settings["draco_decoder"],
                                       position_quantization_bits=int(point["qp"]), prequantize=False,
                                       compression_level=7, skip_decode=False)
        encoder.validate()
        bitstream = folder / "geometry.drc"
        run_command(encoder._encode_command(str(geometry), str(bitstream)), folder, "encode", timeout, cwd=encoder.root)
        run_command(encoder._decode_command(str(bitstream), str(decoded)), folder, "decode", timeout, cwd=encoder.root)
        streams = [bitstream]
        metadata.update(binary_sha256=sha256(encoder.encoder_path), decoder_sha256=sha256(encoder.decoder_path),
                        rate_definition="complete geometry-only Draco point-cloud stream including headers", compression_level=7)
    else:
        request = dict(codec=codec, source=str(geometry), folder=str(folder), point=point, settings=settings)
        save_json(folder / "codec_request.json", request)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(settings["threads"]),
                   MKL_NUM_THREADS=str(settings["threads"]), PYTHONHASHSEED="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        if settings["codec_device"] == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        import json
        python = settings["sparse_python"] if codec == "SparsePCGC" else settings["oa_python"]
        run_command([python, str(Path(__file__).with_name("learned_worker.py")), str(folder / "codec_request.json")],
                    folder, "codec_worker", timeout, cwd=REPO, env=env)
        metadata = json.loads((folder / "codec_result.json").read_text())
        streams = [Path(p) for p in metadata.pop("streams")]
    if not decoded.is_file():
        raise RuntimeError("Codec did not produce a final decoded point cloud")
    reconstructed, _ = read_vertices(decoded)
    if codec in ("G-PCC", "Draco"):
        metadata["native_version_banner"] = (folder / "encode.stdout.txt").read_text().splitlines()[:6]
    bits, artifacts = measure_streams(streams)
    return dict(bits=bits, decoded_path=str(decoded), decoded_points=len(reconstructed),
                decoded_sha256=sha256(decoded), streams=artifacts, codec_metadata=metadata)
