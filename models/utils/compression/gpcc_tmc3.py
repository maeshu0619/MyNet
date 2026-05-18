import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
import torch


class GPCCGeometryEncoder:
    """
    Small G-PCC/tmc3 geometry encoder used by myNet_new training.

    This mirrors the actual tmc3 invocation used by
    compress/octree/G-PCC/encoder_multiple.py without importing or modifying
    that multi-file experiment script.
    """

    codec_name = "gpcc"

    def __init__(
        self,
        root,
        encoder_path,
        cfg_dir,
        tmp_root="",
        timeout=120.0,
        effective_qs=1.0,
        prequantize=True,
        disable_attribute_coding=True,
        merge_duplicated_points=True,
        writer=None,
    ):
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        self.root = self._resolve_path(root or os.path.join(self.repo_root, "compress", "octree", "G-PCC"), self.repo_root)
        self.encoder_path = self._resolve_path(encoder_path or os.path.join(self.root, "build", "tmc3", "tmc3"), self.root)
        default_cfg = os.path.join(
            self.root,
            "cfg",
            "octree-predlift",
            "lossless-geom-lossless-attrs",
            "longdress_vox10_1300",
        )
        cfg_path = self._resolve_path(cfg_dir or default_cfg, self.root)
        if os.path.isdir(cfg_path):
            cfg_path = os.path.join(cfg_path, "encoder.cfg")
        self.cfg_path = cfg_path
        self.tmp_root = str(tmp_root or "")
        self.timeout = max(float(timeout), 1.0)
        self.effective_qs = max(float(effective_qs), 1e-12)
        self.prequantize = bool(prequantize)
        self.disable_attribute_coding = bool(disable_attribute_coding)
        self.merge_duplicated_points = bool(merge_duplicated_points)
        self.writer = writer
        self._request_id = 0
        self._validated = False

    def _resolve_path(self, raw_path, base_dir):
        raw = os.path.expanduser(str(raw_path))
        if os.path.isabs(raw):
            return os.path.abspath(raw)
        candidates = [
            os.path.abspath(os.path.join(os.getcwd(), raw)),
            os.path.abspath(os.path.join(base_dir, raw)),
            os.path.abspath(os.path.join(self.repo_root, raw)),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[1]

    def validate(self):
        if self._validated:
            return
        if not os.path.isfile(self.encoder_path):
            raise FileNotFoundError(f"G-PCC tmc3 encoder not found: {self.encoder_path}")
        if not os.path.isfile(self.cfg_path):
            raise FileNotFoundError(f"G-PCC encoder.cfg not found: {self.cfg_path}")
        self._validated = True
        if self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(
                "G-PCC actual encoder loaded: "
                f"encoder={self.encoder_path}, cfg={self.cfg_path}, "
                f"prequantize={self.prequantize}, effective_qs={self.effective_qs}, "
                f"geometry_only={self.disable_attribute_coding}"
            )

    def _make_tmp_dir(self):
        root = self.tmp_root
        if not root:
            root = "/dev/shm/mynet_gpcc_teacher" if os.path.isdir("/dev/shm") else None
        if root:
            os.makedirs(root, exist_ok=True)
            return tempfile.mkdtemp(prefix="gpcc_actual_", dir=root)
        return tempfile.mkdtemp(prefix="gpcc_actual_")

    def _quantized_coords(self, pts_3n):
        pts = (
            pts_3n.detach()
            .transpose(0, 1)
            .contiguous()
            .to(device="cpu", dtype=torch.float32)
        )
        finite = torch.isfinite(pts).all(dim=1)
        pts = pts[finite]
        if pts.numel() == 0:
            return np.zeros((0, 3), dtype=np.int32)
        if self.prequantize:
            coords = torch.round(pts / self.effective_qs).to(torch.long)
        else:
            coords = torch.round(pts).to(torch.long)
        coords = coords - coords.amin(dim=0, keepdim=True)
        coords = torch.unique(coords, dim=0)
        if coords.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.int32)
        coords_np = coords.to(torch.int32).numpy()
        order = np.lexsort((coords_np[:, 2], coords_np[:, 1], coords_np[:, 0]))
        return coords_np[order]

    @staticmethod
    def _write_ascii_ply(path, coords):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="ascii") as file:
            file.write("ply\n")
            file.write("format ascii 1.0\n")
            file.write(f"element vertex {int(coords.shape[0])}\n")
            file.write("property float x\n")
            file.write("property float y\n")
            file.write("property float z\n")
            file.write("end_header\n")
            for x, y, z in coords:
                file.write(f"{int(x)} {int(y)} {int(z)}\n")

    def _command(self, ply_path, bin_path, rec_path):
        cmd = [
            self.encoder_path,
            "-c",
            self.cfg_path,
            f"--uncompressedDataPath={ply_path}",
            f"--reconstructedDataPath={rec_path}",
            f"--compressedStreamPath={bin_path}",
            "--mode=0",
            "--autoSeqBbox=1",
            "--positionQuantizationScale=1",
            "--trisoupNodeSizeLog2=0",
            f"--mergeDuplicatedPoints={1 if self.merge_duplicated_points else 0}",
        ]
        if self.disable_attribute_coding:
            cmd.append("--disableAttributeCoding=1")
        return cmd

    def encode_tensor(self, pts_3n):
        self.validate()
        encode_t0 = time.time()
        coords = self._quantized_coords(pts_3n)
        point_count = int(coords.shape[0])
        if point_count <= 0:
            return {
                "bit": 0.0,
                "bpp": 0.0,
                "bpn": 0.0,
                "single": 0.0,
                "node": 0.0,
                "point_count": 0,
                "codec": self.codec_name,
                "encode_time": float(time.time() - encode_t0),
            }

        tmp_dir = self._make_tmp_dir()
        try:
            self._request_id += 1
            ply_path = os.path.join(tmp_dir, f"input_{self._request_id}.ply")
            bin_path = os.path.join(tmp_dir, f"encoded_{self._request_id}.bin")
            rec_path = os.path.join(tmp_dir, f"recon_{self._request_id}.ply")
            self._write_ascii_ply(ply_path, coords)
            cmd = self._command(ply_path, bin_path, rec_path)
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout,
                cwd=self.root,
            )
            if result.returncode != 0:
                stdout_tail = (result.stdout or "")[-1200:]
                stderr_tail = (result.stderr or "")[-1200:]
                raise RuntimeError(
                    "G-PCC teacher encode failed "
                    f"(returncode={result.returncode}, cmd={' '.join(cmd)}).\n"
                    f"stdout_tail={stdout_tail}\nstderr_tail={stderr_tail}"
                )
            bit = float(os.path.getsize(bin_path) * 8) if os.path.isfile(bin_path) else 0.0
            bbox_min = coords.amin(dim=0).detach().cpu().tolist() if coords.numel() > 0 else [0, 0, 0]
            bbox_max = coords.amax(dim=0).detach().cpu().tolist() if coords.numel() > 0 else [0, 0, 0]
            return {
                "bit": bit,
                "bpp": bit / max(float(point_count), 1.0),
                "bpn": bit,
                "single": 0.0,
                "node": 0.0,
                "point_count": point_count,
                "codec": self.codec_name,
                "mode": "tmc3_geometry_octree",
                "encode_time": float(time.time() - encode_t0),
                "unique_coord_count": int(point_count),
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "bitstream_bytes": int(os.path.getsize(bin_path)) if os.path.isfile(bin_path) else 0,
            }
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
