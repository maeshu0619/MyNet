import os
import shutil
import subprocess
import tempfile

import numpy as np
import torch


class DracoGeometryEncoder:
    """Small Draco point-cloud encoder used by myNet training."""

    codec_name = "draco"

    def __init__(
        self,
        root,
        encoder_path,
        decoder_path="",
        tmp_root="",
        timeout=120.0,
        effective_qs=1.0,
        prequantize=True,
        position_quantization_bits=0,
        compression_level=7,
        skip_decode=True,
        force_point_cloud=True,
        merge_duplicated_points=True,
        writer=None,
    ):
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        self.root = self._resolve_path(root or os.path.join(self.repo_root, "compress", "octree", "Draco"), self.repo_root)
        self.encoder_path = self._resolve_path(encoder_path or os.path.join(self.root, "build", "draco_encoder"), self.root)
        self.decoder_path = self._resolve_path(decoder_path or os.path.join(self.root, "build", "draco_decoder"), self.root)
        self.tmp_root = str(tmp_root or "")
        self.timeout = max(float(timeout), 1.0)
        self.effective_qs = max(float(effective_qs), 1e-12)
        self.prequantize = bool(prequantize)
        self.position_quantization_bits = int(position_quantization_bits)
        self.compression_level = min(max(int(compression_level), 0), 10)
        self.skip_decode = bool(skip_decode)
        self.force_point_cloud = bool(force_point_cloud)
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
            raise FileNotFoundError(f"Draco encoder not found: {self.encoder_path}")
        if not self.skip_decode and not os.path.isfile(self.decoder_path):
            raise FileNotFoundError(f"Draco decoder not found: {self.decoder_path}")
        self._validated = True
        if self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(
                "Draco actual encoder loaded: "
                f"encoder={self.encoder_path}, prequantize={self.prequantize}, "
                f"effective_qs={self.effective_qs}, qp={self.position_quantization_bits}, "
                f"cl={self.compression_level}, skip_decode={self.skip_decode}"
            )

    def _make_tmp_dir(self):
        root = self.tmp_root
        if not root:
            root = "/dev/shm/mynet_draco_teacher" if os.path.isdir("/dev/shm") else None
        if root:
            os.makedirs(root, exist_ok=True)
            return tempfile.mkdtemp(prefix="draco_actual_", dir=root)
        return tempfile.mkdtemp(prefix="draco_actual_")

    def _prepare_points(self, pts_3n):
        pts = (
            pts_3n.detach()
            .transpose(0, 1)
            .contiguous()
            .to(device="cpu", dtype=torch.float32)
        )
        finite = torch.isfinite(pts).all(dim=1)
        pts = pts[finite]
        if pts.numel() == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if self.prequantize:
            coords = torch.round(pts / self.effective_qs).to(torch.long)
            coords = coords - coords.amin(dim=0, keepdim=True)
            if self.merge_duplicated_points:
                coords = torch.unique(coords, dim=0)
            coords_np = coords.to(torch.int64).numpy()
            order = np.lexsort((coords_np[:, 2], coords_np[:, 1], coords_np[:, 0]))
            return coords_np[order].astype(np.float32, copy=False)
        pts_np = pts.numpy().astype(np.float32, copy=False)
        if self.merge_duplicated_points:
            pts_np = np.unique(pts_np, axis=0)
        return pts_np

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
                file.write(f"{float(x):.9g} {float(y):.9g} {float(z):.9g}\n")

    def _encode_command(self, ply_path, bitstream_path):
        cmd = [self.encoder_path]
        if self.force_point_cloud:
            cmd.append("-point_cloud")
        cmd.extend(
            [
                "-i",
                ply_path,
                "-o",
                bitstream_path,
                "-qp",
                str(int(self.position_quantization_bits)),
                "-cl",
                str(int(self.compression_level)),
            ]
        )
        return cmd

    def _decode_command(self, bitstream_path, decoded_path):
        return [self.decoder_path, "-i", bitstream_path, "-o", decoded_path]

    def encode_tensor(self, pts_3n):
        self.validate()
        coords = self._prepare_points(pts_3n)
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
            }

        tmp_dir = self._make_tmp_dir()
        try:
            self._request_id += 1
            ply_path = os.path.join(tmp_dir, f"input_{self._request_id}.ply")
            bitstream_path = os.path.join(tmp_dir, f"encoded_{self._request_id}.drc")
            decoded_path = os.path.join(tmp_dir, f"decoded_{self._request_id}.ply")
            self._write_ascii_ply(ply_path, coords)
            cmd = self._encode_command(ply_path, bitstream_path)
            enc_start = time_time()
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout,
                cwd=self.root,
            )
            enc_time = time_time() - enc_start
            if result.returncode != 0:
                stdout_tail = (result.stdout or "")[-1200:]
                stderr_tail = (result.stderr or "")[-1200:]
                raise RuntimeError(
                    "Draco teacher encode failed "
                    f"(returncode={result.returncode}, cmd={' '.join(cmd)}).\n"
                    f"stdout_tail={stdout_tail}\nstderr_tail={stderr_tail}"
                )
            dec_time = 0.0
            if not self.skip_decode:
                dec_cmd = self._decode_command(bitstream_path, decoded_path)
                dec_start = time_time()
                dec_result = subprocess.run(
                    dec_cmd,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=self.timeout,
                    cwd=self.root,
                )
                dec_time = time_time() - dec_start
                if dec_result.returncode != 0:
                    stdout_tail = (dec_result.stdout or "")[-1200:]
                    stderr_tail = (dec_result.stderr or "")[-1200:]
                    raise RuntimeError(
                        "Draco teacher decode failed "
                        f"(returncode={dec_result.returncode}, cmd={' '.join(dec_cmd)}).\n"
                        f"stdout_tail={stdout_tail}\nstderr_tail={stderr_tail}"
                    )
            bit = float(os.path.getsize(bitstream_path) * 8) if os.path.isfile(bitstream_path) else 0.0
            return {
                "bit": bit,
                "bpp": bit / max(float(point_count), 1.0),
                "bpn": bit,
                "single": 0.0,
                "node": 0.0,
                "point_count": point_count,
                "codec": self.codec_name,
                "mode": "draco_point_cloud",
                "qp": int(self.position_quantization_bits),
                "cl": int(self.compression_level),
                "enc_time": float(enc_time),
                "dec_time": float(dec_time),
            }
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def time_time():
    import time

    return time.time()
