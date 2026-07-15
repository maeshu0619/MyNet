import atexit
import hashlib
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import subprocess
from collections import OrderedDict

import numpy as np
import torch

from models.utils.compression.gpcc_tmc3 import GPCCGeometryEncoder
from models.utils.compression.draco import DracoGeometryEncoder
from models.utils.compression.proxy_octree import ProxyOctreeConfig, SoftOctreeRateProxy


class _OctAttentionActualEncoder:
    """
    Run the same OctAttention path as compress/octree/OctAttention/encoder.py.

    The default path stays out of autograd and estimates the hard OctAttention
    bit cost fully in memory.  The legacy disk-backed path is kept only for the
    arithmetic-coding mode.
    """
    def __init__(self, args, writer=None):
        self.args = args
        self.writer = writer
        self.qs = float(getattr(args, "qs", 2.0))
        self.actualcode = bool(getattr(args, "octattention_actualcode", False))
        self.tmp_root = getattr(args, "octattention_tmp_dir", "")
        self._loaded = False
        self._model = None
        self._data_prepare = None
        self._write_ply_data = None
        self._oa_bptt = int(getattr(args, "bptt", 1024))
        self._fast_proxy = None

    def _resolve_ckpt_path(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        oa_dir = os.path.join(repo_root, "compress", "octree", "OctAttention")
        ckpt_path = getattr(self.args, "octattention_ckpt", "")
        if not ckpt_path:
            ckpt_path = os.path.join(oa_dir, "modelsave", "obj", "encoder_epoch_00800093.pth")
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.abspath(os.path.join(repo_root, ckpt_path))
        return repo_root, oa_dir, ckpt_path

    def _lazy_init(self):
        if self._loaded:
            return

        repo_root, oa_dir, ckpt_path = self._resolve_ckpt_path()
        teacher_mode = str(getattr(self.args, "octattention_teacher_device", "balanced")).strip().lower()

        if not self.actualcode:
            self._fast_proxy = SoftOctreeRateProxy(
                ProxyOctreeConfig(
                    max_depth=12,
                    qs=self.qs,
                    bptt=self._oa_bptt,
                    checkpoint_path=ckpt_path,
                    teacher_device=teacher_mode,
                )
            )
            self._loaded = True
            if self.writer is not None and hasattr(self.writer, "write"):
                self.writer.write(
                    f"OctAttention actual encoder loaded: {ckpt_path} "
                    f"(backend=in_memory, teacher_device={teacher_mode})"
                )
            return

        project_root = os.path.join(repo_root, "myNet")
        if project_root not in sys.path:
            sys.path.append(project_root)
        if oa_dir not in sys.path:
            sys.path.append(oa_dir)

        from Preparedata.data import dataPrepare
        from networkTool import levelNumK
        from pt import write_ply_data
        from models.utils.compression.proxy_octree import _OctAttentionTeacherModel

        save_dic = torch.load(ckpt_path, map_location="cpu")
        state_dict = save_dic["encoder"] if "encoder" in save_dic else save_dic

        oa_model = _OctAttentionTeacherModel(max_octree_level=12)
        oa_model.load_state_dict(state_dict)
        if teacher_mode == "cpu":
            oa_device = torch.device("cpu")
        elif teacher_mode == "balanced":
            oa_device = torch.device("cpu")
        elif teacher_mode.startswith("cuda") and torch.cuda.is_available():
            oa_device = torch.device(teacher_mode if teacher_mode != "cuda" else "cuda")
        else:
            oa_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        oa_model.to(oa_device)
        oa_model.eval()
        for param in oa_model.parameters():
            param.requires_grad_(False)

        self._model = oa_model
        self._data_prepare = dataPrepare
        self._level_num_k = levelNumK
        self._write_ply_data = write_ply_data
        self._oa_device = oa_device
        self._loaded = True

        if self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(f"OctAttention actual encoder loaded: {ckpt_path} (device={oa_device})")

    def _make_tmp_dir(self):
        root = self.tmp_root
        if not root:
            root = "/dev/shm/mynet_octattention_actual" if os.path.isdir("/dev/shm") else None
        if root:
            os.makedirs(root, exist_ok=True)
            return tempfile.mkdtemp(prefix="oa_actual_", dir=root)
        return tempfile.mkdtemp(prefix="oa_actual_")

    def encode_bits(self, pts_3n):
        self._lazy_init()
        if self._fast_proxy is not None:
            with torch.inference_mode():
                pts = pts_3n.to(torch.float32)
                point_w = torch.ones((pts.shape[-1],), device=pts.device, dtype=pts.dtype)
                prepared = self._fast_proxy._prepare_single_hard_octattention_eval(
                    pts,
                    point_w,
                    qs_value=self.qs,
                )
                if prepared is None:
                    return {
                        "bit": 0.0,
                        "bpp": 0.0,
                        "bpn": 0.0,
                        "single": 0.0,
                        "node": 0.0,
                        "point_count": int(pts.shape[-1]),
                    }
                _safe_pts, _valid, _offset_np, _max_level, oct_seq_np, _teacher_log2, hard_bits, hard_node, hard_single = prepared
                bit = float(hard_bits.detach())
                node = float(hard_node.detach())
                single = float(hard_single.detach())
                point_count = int(pts.shape[-1])
                return {
                    "bit": bit,
                    "bpp": bit / max(float(point_count), 1.0),
                    "bpn": bit / max(node, 1.0),
                    "single": single,
                    "node": node,
                    "point_count": point_count,
                    "oct_len": int(oct_seq_np.shape[0]),
                }

        pts_np = (
            pts_3n.detach()
            .transpose(0, 1)
            .contiguous()
            .to("cpu")
            .numpy()
            .astype(np.float32, copy=False)
        )

        tmp_dir = self._make_tmp_dir()
        try:
            ply_path = os.path.join(tmp_dir, "input.ply")
            mat_dir = os.path.join(tmp_dir, "mat")
            bin_path = os.path.join(tmp_dir, "encoded.bin")
            self._write_ply_data(ply_path, pts_np)
            mat_file, _dq_pt, _ref_pt = self._data_prepare(
                ply_path,
                saveMatDir=mat_dir,
                qs=self.qs,
                ptNamePrefix="",
                rotation=False,
            )

            import h5py as _h5py

            mat = None
            try:
                mat = _h5py.File(mat_file, "r")
                cell = mat["patchFile"]
                ref = cell[0, 0]
                data_arr = np.array(mat[ref])
                oct_data_seq = np.transpose(data_arr).astype(np.int32)[:, -self._level_num_k:, 0:6]
            finally:
                if mat is not None:
                    mat.close()

            binsz, oct_len = self._compress_oct_seq(oct_data_seq, bin_path)
            single_count = self._single_child_count(oct_data_seq)
            return {
                "bit": float(binsz),
                "bpp": float(binsz) / max(float(pts_3n.shape[-1]), 1.0),
                "bpn": float(binsz) / max(float(oct_len), 1.0),
                "single": float(single_count),
                "node": float(oct_len),
                "point_count": int(pts_3n.shape[-1]),
            }
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def encode_bits_many(self, candidates, *, max_parallel=1, mode="single", fallback_to_single=True):
        return _encode_bits_many_sequential(
            self,
            candidates,
            max_parallel=max_parallel,
            mode=mode,
            fallback_to_single=fallback_to_single,
        )

    @staticmethod
    def _single_child_count(oct_data_seq):
        oct_code = oct_data_seq[:, -1, 0].astype(np.int32)
        pop = (
            (oct_code & 1) + ((oct_code >> 1) & 1) +
            ((oct_code >> 2) & 1) + ((oct_code >> 3) & 1) +
            ((oct_code >> 4) & 1) + ((oct_code >> 5) & 1) +
            ((oct_code >> 6) & 1) + ((oct_code >> 7) & 1)
        )
        return int((pop == 1).sum())

    def _generate_square_subsequent_mask(self, size):
        return torch.triu(
            torch.full((size, size), float("-inf"), device=self._oa_device),
            diagonal=1,
        )

    @staticmethod
    def _batchify(oct_seq, bptt, oct_len):
        oct_seq = oct_seq.copy()
        oct_seq[:-1, 0:-1, :] = oct_seq[1:, 0:-1, :]
        oct_seq[:-1, -1, 1:3] = oct_seq[1:, -1, 1:3]
        oct_seq[:, :, 0] = oct_seq[:, :, 0] - 1
        pad_len = bptt
        padded = np.zeros((bptt + oct_len + pad_len, *oct_seq.shape[1:]), dtype=oct_seq.dtype)
        padded[bptt:bptt + oct_len] = oct_seq
        oct_seq_t = torch.from_numpy(padded).long()

        data_id = torch.full((bptt + oct_len + pad_len,), -1, dtype=torch.long)
        data_id[bptt:bptt + oct_len] = torch.arange(oct_len, dtype=torch.long)
        return data_id.unsqueeze(1), oct_seq_t.unsqueeze(1)

    @staticmethod
    def _estimate_bits(pro_bit, oct_seq, level_id):
        oct_values = oct_seq.astype(np.int64).reshape(-1) - 1
        level_values = np.asarray(level_id, dtype=np.int64).reshape(-1)

        prob_hit = np.take_along_axis(pro_bit[:oct_values.shape[0]], oct_values[:, None], axis=1).squeeze(1)
        bit_each = -np.log2(prob_hit + 1e-7)
        bit = float(bit_each.sum())

        level_change = np.empty(level_values.shape[0], dtype=bool)
        level_change[0] = level_values[0] != 1
        level_change[1:] = level_values[1:] != level_values[:-1]
        level_change_idx = np.flatnonzero(level_change)

        binsz_list = np.concatenate([np.cumsum(bit_each)[level_change_idx], np.array([bit])])
        oct_num_list = np.concatenate([level_change_idx + 1, np.array([oct_values.shape[0]])])
        return bit, binsz_list, oct_num_list

    def _compress_oct_seq(self, oct_data_seq, output_file):
        from networkTool import MAX_OCTREE_LEVEL

        level_id = oct_data_seq[:, -1, 1].copy()
        oct_data_seq = oct_data_seq.copy()
        if level_id.max() > MAX_OCTREE_LEVEL:
            level_id = np.minimum(level_id, MAX_OCTREE_LEVEL)

        oct_seq = oct_data_seq[:, -1:, 0].astype(int)
        oct_len = len(oct_seq)
        bptt_eff = min(self._oa_bptt, oct_len - 1)
        if bptt_eff < 32:
            raise ValueError(f"oct_len too small for OctAttention: oct_len={oct_len}")

        data_id, padded_data = self._batchify(oct_data_seq, bptt_eff, oct_len)
        pading_length = padded_data.shape[0]
        src_mask = self._generate_square_subsequent_mask(bptt_eff)
        pro_bit_chunks = [] if self.actualcode else None
        oct_values = torch.from_numpy(oct_seq.astype(np.int64).reshape(-1) - 1)
        processed = 0
        total_bits = 0.0

        with torch.inference_mode():
            for i in range(0, pading_length - bptt_eff, bptt_eff):
                inp = padded_data[i:i + bptt_eff].to(device=self._oa_device, non_blocking=True)
                node_id = data_id[i + 1:i + bptt_eff + 1].reshape(-1)
                valid_mask = node_id >= 0
                if not valid_mask.any():
                    continue
                output = self._model(inp, src_mask, [])
                output = output.reshape(-1, 255)
                prob = torch.softmax(output, dim=1)
                valid_prob = prob[valid_mask]
                valid_count = min(int(valid_prob.shape[0]), oct_len - processed)
                if valid_count <= 0:
                    break
                valid_prob = valid_prob[:valid_count]
                target = oct_values[processed:processed + valid_count].to(device=self._oa_device, non_blocking=True)
                prob_hit = valid_prob.gather(1, target.view(-1, 1)).squeeze(1)
                total_bits += float((-torch.log2(prob_hit + 1e-7)).sum().detach().cpu())
                if pro_bit_chunks is not None:
                    pro_bit_chunks.append(valid_prob.detach().cpu().numpy())
                processed += valid_count

        if processed <= 0:
            raise ValueError(f"OctAttention produced no valid probability rows: oct_len={oct_len}")
        if processed < oct_len and self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(
                f"OctAttention warning: probability rows shorter than octree length "
                f"({processed}/{oct_len}); using available rows for bit estimate."
            )
        binsz = total_bits

        if self.actualcode:
            import numpyAc
            if not pro_bit_chunks:
                raise ValueError(f"OctAttention produced no probability chunks for arithmetic coding: oct_len={oct_len}")
            pro_bit = np.vstack(pro_bit_chunks)[:oct_len]
            binsz, _binsz_list, _oct_num_list = self._estimate_bits(pro_bit, oct_seq, level_id)
            codec = numpyAc.arithmeticCoding()
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            _, binsz = codec.encode(pro_bit[:oct_len, :], oct_seq.astype(np.int16).squeeze(-1) - 1, output_file)
            del pro_bit
        del data_id, padded_data, src_mask
        if self._oa_device.type == "cuda":
            torch.cuda.empty_cache()
        return float(binsz), int(oct_len)


class _SparsePCGCActualEncoder:
    """
    Persistent SparsePCGC teacher running in the sparsepcgc virtual environment.

    SparsePCGC depends on MinkowskiEngine/PyTorch versions that differ from the
    myOA training environment, so the teacher is isolated behind a JSON-lines
    worker while keeping the same encode_bits() contract as OctAttention.
    """

    codec_name = "sparsepcgc"

    def __init__(self, args, writer=None):
        self.args = args
        self.writer = writer
        self.tmp_root = getattr(args, "sparsepcgc_tmp_dir", "")
        self.timeout = float(getattr(args, "sparsepcgc_timeout", 600.0))
        self._proc = None
        self._stderr_file = None
        self._stderr_path = None
        self._request_id = 0
        self._loaded = False
        self._stdout_queue = None
        self._stdout_thread = None
        # SparsePCGC workerは逐次requestで使うため、一時workspaceを再利用する。
        # 毎Stepのmkdir/rmtreeを避けるが、入力PLYと出力bitstreamはrequestごとに上書きする。
        self._workspace_dir = None
        self._workspace_lock = threading.Lock()
        self._result_cache = OrderedDict()
        self._codec_fingerprint = None

    def _repo_root(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

    def _sparsepcgc_root(self):
        root = getattr(self.args, "sparsepcgc_root", "")
        if not root:
            root = os.path.join(self._repo_root(), "compress", "octree", "SparsePCGC")
        if not os.path.isabs(root):
            root = os.path.abspath(os.path.join(self._repo_root(), root))
        return root

    def _python_command(self):
        explicit = str(getattr(self.args, "sparsepcgc_python", "")).strip()
        if explicit:
            return [explicit]

        env_name = str(getattr(self.args, "sparsepcgc_env", "sparsepcgc")).strip() or "sparsepcgc"
        candidates = []
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            envs_dir = os.path.dirname(conda_prefix)
            candidates.append(os.path.join(envs_dir, env_name, "bin", "python"))
        candidates.extend(
            [
                os.path.expanduser(os.path.join("~", "miniconda3", "envs", env_name, "bin", "python")),
                os.path.expanduser(os.path.join("~", "anaconda3", "envs", env_name, "bin", "python")),
            ]
        )
        for candidate in candidates:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return [candidate]
        return ["conda", "run", "-n", env_name, "python"]

    def _make_tmp_dir(self):
        root = self.tmp_root
        if not root:
            root = "/dev/shm/mynet_sparsepcgc_teacher" if os.path.isdir("/dev/shm") else None
        if root:
            os.makedirs(root, exist_ok=True)
            return tempfile.mkdtemp(prefix="spgc_actual_", dir=root)
        return tempfile.mkdtemp(prefix="spgc_actual_")

    @staticmethod
    def _csv_arg(value):
        if isinstance(value, (list, tuple)):
            return ",".join(str(int(item)) for item in value)
        return str(value)

    def _worker_args(self):
        repo_root = self._repo_root()
        runner = os.path.join(repo_root, "myNet", "models", "utils", "loss", "sparsepcgc_teacher_worker.py")
        root = self._sparsepcgc_root()
        cmd = self._python_command() + [
            runner,
            "--sparsepcgc-root",
            root,
            "--mode",
            str(getattr(self.args, "sparsepcgc_mode", "dense_lossless")),
            "--device",
            str(getattr(self.args, "sparsepcgc_device", "auto")),
            "--ckptdir",
            str(getattr(self.args, "sparsepcgc_ckptdir", os.path.join(root, "ckpts", "dense", "epoch_last.pth"))),
            "--ckptdir-sr",
            str(getattr(self.args, "sparsepcgc_ckptdir_sr", os.path.join(root, "ckpts", "dense_1stage", "epoch_last.pth"))),
            "--ckptdir-ae",
            str(getattr(self.args, "sparsepcgc_ckptdir_ae", os.path.join(root, "ckpts", "dense_slne", "epoch_last.pth"))),
            "--ckptdir-low",
            str(getattr(self.args, "sparsepcgc_ckptdir_low", os.path.join(root, "ckpts", "sparse_low", "epoch_last.pth"))),
            "--ckptdir-high",
            str(getattr(self.args, "sparsepcgc_ckptdir_high", os.path.join(root, "ckpts", "sparse_high", "epoch_last.pth"))),
            "--ckptdir-offset",
            str(getattr(self.args, "sparsepcgc_ckptdir_offset", os.path.join(root, "ckpts", "sparse_offset", "epoch_last.pth"))),
            "--voxel-size",
            str(float(getattr(self.args, "sparsepcgc_voxel_size", 1.0))),
            "--pos-quantscale",
            str(int(getattr(self.args, "sparsepcgc_pos_quantscale", 1))),
            "--psnr-resolution",
            str(int(getattr(self.args, "sparsepcgc_psnr_resolution", 1023))),
            "--dense-scale-ae-list",
            self._csv_arg(getattr(self.args, "sparsepcgc_dense_scale_ae_list", "1,0,1,0,1,0")),
            "--dense-scale-sr-list",
            self._csv_arg(getattr(self.args, "sparsepcgc_dense_scale_sr_list", "0,1,1,2,2,3")),
            "--pos-quantscale-list",
            self._csv_arg(getattr(self.args, "sparsepcgc_pos_quantscale_list", "4")),
            "--scale-m",
            str(int(getattr(self.args, "sparsepcgc_scale_m", 8))),
            "--scale-ae",
            str(int(getattr(self.args, "sparsepcgc_scale_ae", 0))),
            "--scale-sr",
            str(int(getattr(self.args, "sparsepcgc_scale_sr", 2))),
        ]
        if bool(getattr(self.args, "sparsepcgc_inner_psnr", False)):
            cmd.append("--inner-psnr")
        if bool(getattr(self.args, "sparsepcgc_worker_gpu_stats", False)):
            cmd.append("--gpu-stats")
        if bool(getattr(self.args, "sparsepcgc_worker_gpu_stats_print", False)):
            cmd.append("--gpu-stats-print")
        if bool(getattr(self.args, "sparsepcgc_offset", False)):
            cmd.append("--offset")
        if bool(getattr(self.args, "sparsepcgc_test_d2", False)):
            cmd.append("--test-d2")
        if not bool(getattr(self.args, "sparsepcgc_skip_decode", True)):
            cmd.append("--decode")
        return cmd, root

    def _direct_env_prefix(self, cmd):
        if not cmd or not os.path.isabs(cmd[0]):
            return None
        marker = os.path.join("envs", str(getattr(self.args, "sparsepcgc_env", "sparsepcgc")), "bin", "python")
        if cmd[0].endswith(marker):
            return os.path.dirname(os.path.dirname(cmd[0]))
        return None

    def _lazy_init(self):
        if self._loaded:
            return

        cmd, sparse_root = self._worker_args()
        tmp_root = self.tmp_root or ("/dev/shm/mynet_sparsepcgc_teacher" if os.path.isdir("/dev/shm") else tempfile.gettempdir())
        os.makedirs(tmp_root, exist_ok=True)
        self._stderr_path = os.path.join(tmp_root, f"sparsepcgc_teacher_{os.getpid()}.stderr.log")
        self._stderr_file = open(self._stderr_path, "a", encoding="utf-8", buffering=1)
        self._stderr_file.write("CMD: " + " ".join(cmd) + "\n")

        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", str(max(int(getattr(self.args, "sparsepcgc_omp_threads", 12)), 1)))
        env_prefix = self._direct_env_prefix(cmd)
        if env_prefix is not None:
            env["CONDA_PREFIX"] = env_prefix
            env["PATH"] = os.path.join(env_prefix, "bin") + os.pathsep + env.get("PATH", "")

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,
            cwd=sparse_root,
            env=env,
        )
        self._stdout_queue = queue.Queue()
        self._stdout_thread = threading.Thread(target=self._pump_stdout, daemon=True)
        self._stdout_thread.start()
        atexit.register(self.close)
        ready = self._read_worker_json(self.timeout)
        if ready.get("status") != "ready":
            raise RuntimeError(f"SparsePCGC teacher failed to initialize: {ready}")
        self._loaded = True

        if self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(
                "SparsePCGC actual encoder loaded: "
                f"mode={getattr(self.args, 'sparsepcgc_mode', 'dense_lossy')}, "
                f"m={int(getattr(self.args, 'sparsepcgc_scale_m', 8))}, "
                f"AE={int(getattr(self.args, 'sparsepcgc_scale_ae', 0))}, "
                f"SR={int(getattr(self.args, 'sparsepcgc_scale_sr', 2))}, "
                f"device={getattr(self.args, 'sparsepcgc_device', 'auto')}, "
                f"python={' '.join(self._python_command())}, stderr={self._stderr_path}"
            )

    def _pump_stdout(self):
        proc = self._proc
        if proc is None or proc.stdout is None or self._stdout_queue is None:
            return
        try:
            for line in proc.stdout:
                self._stdout_queue.put(line)
        except Exception as exc:
            self._stdout_queue.put(json.dumps({"status": "stdout_error", "message": str(exc)}) + "\n")

    def _read_worker_json(self, timeout):
        if self._proc is None or self._stdout_queue is None:
            raise RuntimeError("SparsePCGC worker is not running.")
        deadline = time.time() + max(float(timeout), 1.0)
        while time.time() < deadline:
            if self._proc.poll() is not None and self._stdout_queue.empty():
                raise RuntimeError(
                    f"SparsePCGC worker exited with code {self._proc.returncode}. "
                    f"See stderr log: {self._stderr_path}"
                )
            remaining = max(deadline - time.time(), 0.0)
            try:
                line = self._stdout_queue.get(timeout=min(0.5, remaining))
            except queue.Empty:
                continue
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                if self.writer is not None and hasattr(self.writer, "write"):
                    self.writer.write(f"SparsePCGC teacher non-json stdout: {line.strip()}")
        raise TimeoutError(f"SparsePCGC worker timed out after {timeout}s. See stderr log: {self._stderr_path}")

    def _send_worker_request(self, request):
        """Persistent workerへ1 requestを送り、対応するresponseだけを返す。"""
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("SparsePCGC worker stdin is not available.")
        self._request_id += 1
        request = dict(request)
        request["request_id"] = self._request_id
        start = time.time()
        self._proc.stdin.write(json.dumps(request, sort_keys=True) + "\n")
        self._proc.stdin.flush()
        response = self._read_worker_json(self.timeout)
        while response.get("request_id") not in {None, self._request_id}:
            response = self._read_worker_json(self.timeout)
        return response, float(time.time() - start)

    def _write_ply(self, path, pts_3n):
        """旧private API互換。既存呼出元のASCII PLY挙動を維持する。"""
        from models.utils.io.utils_ply import write_ply

        pts_np = (
            pts_3n.detach()
            .transpose(0, 1)
            .contiguous()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
        )
        ok = write_ply(path, pts_np, ["x", "y", "z"])
        if not ok:
            raise RuntimeError(f"Failed to write SparsePCGC teacher PLY: {path}")

    def _workspace(self):
        if self._workspace_dir is not None:
            return self._workspace_dir
        root = self.tmp_root
        if not root:
            root = "/dev/shm/mynet_sparsepcgc_teacher" if os.path.isdir("/dev/shm") else None
        if root:
            os.makedirs(root, exist_ok=True)
            self._workspace_dir = tempfile.mkdtemp(prefix="spgc_actual_worker_", dir=root)
        else:
            self._workspace_dir = tempfile.mkdtemp(prefix="spgc_actual_worker_")
        return self._workspace_dir

    @staticmethod
    def _points_numpy_and_hash(pts_3n):
        transfer_start = time.time()
        pts_np = (
            pts_3n.detach()
            .transpose(0, 1)
            .contiguous()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
        )
        pts_np = np.ascontiguousarray(pts_np, dtype=np.float32)
        digest = hashlib.sha256()
        digest.update(str(tuple(pts_np.shape)).encode("ascii"))
        digest.update(memoryview(pts_np).cast("B"))
        return pts_np, digest.hexdigest(), float(time.time() - transfer_start)

    def _write_ply_array(self, path, pts_np, *, binary=None):
        write_start = time.time()
        if binary is None:
            binary = bool(getattr(self.args, "sparsepcgc_fast_binary_ply", True))
        if bool(binary):
            # 座標値は従来と同じfloat32であり、変えるのはPLYの保存形式だけである。
            # binary little-endianにすることでASCII文字列化の時間と容量を削減する。
            arr = np.asarray(pts_np, dtype="<f4", order="C")
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {int(arr.shape[0])}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "end_header\n"
            ).encode("ascii")
            with open(path, "wb", buffering=1024 * 1024) as handle:
                handle.write(header)
                arr.tofile(handle)
        else:
            from models.utils.io.utils_ply import write_ply
            ok = write_ply(path, pts_np, ["x", "y", "z"])
            if not ok:
                raise RuntimeError(f"Failed to write SparsePCGC teacher PLY: {path}")
        return float(time.time() - write_start)

    def _codec_cache_fingerprint(self):
        """実測bitを変えうるcodec条件をLRU keyへ固定する。"""
        if self._codec_fingerprint is not None:
            return self._codec_fingerprint
        checkpoint_fields = (
            "sparsepcgc_ckptdir",
            "sparsepcgc_ckptdir_sr",
            "sparsepcgc_ckptdir_ae",
            "sparsepcgc_ckptdir_low",
            "sparsepcgc_ckptdir_high",
            "sparsepcgc_ckptdir_offset",
        )
        checkpoints = {}
        for field in checkpoint_fields:
            path = os.path.abspath(os.path.expanduser(str(getattr(self.args, field, ""))))
            try:
                stat = os.stat(path)
                checkpoints[field] = {"path": path, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
            except OSError:
                checkpoints[field] = {"path": path, "missing": True}
        payload = {
            "root": os.path.abspath(os.path.expanduser(str(getattr(self.args, "sparsepcgc_root", "")))),
            "mode": str(getattr(self.args, "sparsepcgc_mode", "dense_lossy")),
            "m": int(getattr(self.args, "sparsepcgc_scale_m", 8)),
            "ae": int(getattr(self.args, "sparsepcgc_scale_ae", 0)),
            "sr": int(getattr(self.args, "sparsepcgc_scale_sr", 2)),
            "voxel_size": float(getattr(self.args, "sparsepcgc_voxel_size", 1.0)),
            "pos_quantscale": int(getattr(self.args, "sparsepcgc_pos_quantscale", 1)),
            "psnr_resolution": int(getattr(self.args, "sparsepcgc_psnr_resolution", 1023)),
            "dense_scale_ae_list": str(getattr(self.args, "sparsepcgc_dense_scale_ae_list", "")),
            "dense_scale_sr_list": str(getattr(self.args, "sparsepcgc_dense_scale_sr_list", "")),
            "checkpoints": checkpoints,
        }
        self._codec_fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return self._codec_fingerprint

    def _result_cache_key(
        self,
        point_hash,
        exact_occupancy,
        exact_teacher_mode,
        *,
        occupancy_debug=False,
        occupancy_low_prob_threshold=0.1,
        exact_teacher_uses_full_context=False,
        exact_teacher_fallback_reason="",
    ):
        # bitだけでなくexact occupancy教師もcache対象であるため、教師内容を変える条件を全て含める。
        return "|".join(
            [
                str(point_hash),
                self._codec_cache_fingerprint(),
                str(getattr(self.args, "sparsepcgc_mode", "dense_lossy")),
                str(int(getattr(self.args, "sparsepcgc_scale_m", 8))),
                str(int(getattr(self.args, "sparsepcgc_scale_ae", 0))),
                str(int(getattr(self.args, "sparsepcgc_scale_sr", 2))),
                str(float(getattr(self.args, "sparsepcgc_voxel_size", 1.0))),
                str(int(getattr(self.args, "sparsepcgc_pos_quantscale", 1))),
                str(bool(exact_occupancy)),
                str(exact_teacher_mode),
                str(bool(occupancy_debug)),
                f"{float(occupancy_low_prob_threshold):.12g}",
                str(bool(exact_teacher_uses_full_context)),
                str(exact_teacher_fallback_reason),
            ]
        )

    @staticmethod
    def _looks_like_ply_reader_error(response):
        message = str(response.get("message", response)).lower()
        markers = (
            "ply",
            "vertex",
            "binary_little_endian",
            "read_coords",
            "read_ply",
            "load_data",
            "unable to read",
            "failed to read",
            "parse",
        )
        return any(marker in message for marker in markers)

    def _get_cached_result(self, key):
        if not bool(getattr(self.args, "sparsepcgc_actual_result_cache", True)):
            return None
        item = self._result_cache.get(key)
        if item is None:
            return None
        self._result_cache.move_to_end(key)
        out = dict(item)
        out["sparsepcgc_actual_result_cache_hit"] = True
        return out

    def _store_cached_result(self, key, value):
        if not bool(getattr(self.args, "sparsepcgc_actual_result_cache", True)):
            return
        clean = dict(value)
        clean.pop("bitstream_path", None)
        clean.pop("decoded_path", None)
        clean.pop("decoded_copy_path", None)
        clean["sparsepcgc_actual_result_cache_hit"] = False
        self._result_cache[key] = clean
        self._result_cache.move_to_end(key)
        max_entries = max(int(getattr(self.args, "sparsepcgc_actual_result_cache_max_entries", 256)), 1)
        while len(self._result_cache) > max_entries:
            self._result_cache.popitem(last=False)

    def _exact_occupancy_enabled_this_step(self):
        if not bool(getattr(self.args, "enable_sparsepcgc_exact_occupancy_teacher", False)):
            return False
        interval = int(getattr(self.args, "sparsepcgc_exact_occupancy_interval", 1))
        if interval <= 0:
            return False
        step = int(getattr(self.args, "_global_train_step", 0)) + 1
        return interval <= 1 or (step % interval) == 0

    def encode_bits(self, pts_3n):
        self._lazy_init()
        encode_t0 = time.time()
        with self._workspace_lock:
            workspace = self._workspace() if bool(getattr(self.args, "sparsepcgc_reuse_workspace", True)) else self._make_tmp_dir()
            remove_workspace_after = workspace != self._workspace_dir
            ply_path = os.path.join(workspace, "input.ply")
            output_dir = os.path.join(workspace, "encoded")

            pts_np, point_hash, input_prepare_time = self._points_numpy_and_hash(pts_3n)
            exact_occupancy = bool(self._exact_occupancy_enabled_this_step())
            exact_teacher_mode = str(
                getattr(
                    self.args,
                    "_current_exact_teacher_mode",
                    getattr(self.args, "sparsepcgc_exact_teacher_mode", "auto"),
                )
            )
            occupancy_debug = bool(
                getattr(self.args, "enable_sparsepcgc_occupancy_debug", False) or exact_occupancy
            )
            occupancy_low_prob_threshold = float(
                getattr(self.args, "sparsepcgc_occupancy_low_prob_threshold", 0.1)
            )
            exact_teacher_uses_full_context = bool(
                getattr(self.args, "_current_exact_teacher_uses_full_context", False)
            )
            exact_teacher_fallback_reason = str(
                getattr(self.args, "_current_exact_teacher_fallback_reason", "")
            )
            result_cache_key = self._result_cache_key(
                point_hash,
                exact_occupancy,
                exact_teacher_mode,
                occupancy_debug=occupancy_debug,
                occupancy_low_prob_threshold=occupancy_low_prob_threshold,
                exact_teacher_uses_full_context=exact_teacher_uses_full_context,
                exact_teacher_fallback_reason=exact_teacher_fallback_reason,
            )
            cached = self._get_cached_result(result_cache_key)
            if cached is not None:
                cached["encode_time"] = float(time.time() - encode_t0)
                cached["sparsepcgc_input_prepare_time"] = float(input_prepare_time)
                cached["sparsepcgc_ply_write_time"] = 0.0
                cached["sparsepcgc_worker_roundtrip_time"] = 0.0
                if remove_workspace_after:
                    shutil.rmtree(workspace, ignore_errors=True)
                return cached

            if os.path.isdir(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
            os.makedirs(output_dir, exist_ok=True)
            used_binary_ply = bool(getattr(self.args, "sparsepcgc_fast_binary_ply", True))
            ply_write_time = self._write_ply_array(ply_path, pts_np, binary=used_binary_ply)
            request = {
                "input_file": ply_path,
                "output_dir": output_dir,
                "occupancy_debug": occupancy_debug,
                "occupancy_low_prob_threshold": occupancy_low_prob_threshold,
                "exact_occupancy": bool(exact_occupancy),
                "exact_teacher_mode": exact_teacher_mode,
                "exact_teacher_uses_full_context": exact_teacher_uses_full_context,
                "exact_teacher_fallback_reason": exact_teacher_fallback_reason,
            }
            response, worker_roundtrip_time = self._send_worker_request(request)
            ascii_fallback_used = False
            ascii_fallback_write_time = 0.0
            if (
                response.get("status") != "ok"
                and used_binary_ply
                and bool(getattr(self.args, "sparsepcgc_binary_ply_fallback_ascii", True))
                and self._looks_like_ply_reader_error(response)
            ):
                # reader互換性だけが原因の場合に限り、同じfloat32座標をASCIIへ書き直して再試行する。
                # worker/modelは再起動しないため、失敗時の追加コストを最小限にする。
                ascii_fallback_write_time = self._write_ply_array(ply_path, pts_np, binary=False)
                retry_response, retry_time = self._send_worker_request(request)
                worker_roundtrip_time += float(retry_time)
                response = retry_response
                ascii_fallback_used = True
            if response.get("status") != "ok":
                if remove_workspace_after:
                    shutil.rmtree(workspace, ignore_errors=True)
                raise RuntimeError(
                    "SparsePCGC teacher encode failed: "
                    f"{response.get('message', response)}. See stderr log: {self._stderr_path}"
                )
            result = response.get("result", {})
            result["sparsepcgc_scale_m"] = int(getattr(self.args, "sparsepcgc_scale_m", 8))
            result["sparsepcgc_scale_ae"] = int(getattr(self.args, "sparsepcgc_scale_ae", 0))
            result["sparsepcgc_scale_sr"] = int(getattr(self.args, "sparsepcgc_scale_sr", 2))
            result["sparsepcgc_mode_effective"] = str(getattr(self.args, "sparsepcgc_mode", "dense_lossy"))
            decoded_copy_dir = str(getattr(self.args, "sparsepcgc_decoded_copy_dir", "") or "").strip()
            if decoded_copy_dir:
                decoded_path_text = str(result.get("decoded_path", "") or "")
                decoded_candidates = [
                    item.strip()
                    for item in decoded_path_text.split(",")
                    if item.strip() and item.strip() != "(decode skipped)"
                ]
                copied_paths = []
                os.makedirs(decoded_copy_dir, exist_ok=True)
                for idx, decoded_path in enumerate(decoded_candidates):
                    if not os.path.exists(decoded_path):
                        continue
                    stem = os.path.splitext(os.path.basename(decoded_path))[0]
                    copy_path = os.path.join(
                        decoded_copy_dir,
                        f"request_{self._request_id:06d}_{idx}_{stem}.ply",
                    )
                    shutil.copy2(decoded_path, copy_path)
                    copied_paths.append(copy_path)
                if copied_paths:
                    result["decoded_copy_path"] = ", ".join(copied_paths)
            bit = float(result.get("file_size", result.get("bit", 0.0)))
            point_count = int(result.get("point_count", result.get("num_points_raw", pts_3n.shape[-1])))
            # ============================================================
            # Phase2:
            # SparsePCGC exact occupancy teacher のvalid判定
            # ============================================================
            exact_candidate_count = int(result.get("sparsepcgc_exact_candidate_count", 0) or 0)
            exact_nll = result.get("sparsepcgc_exact_occupancy_nll", float("nan"))
            exact_bits = result.get("sparsepcgc_exact_estimated_bits", float("nan"))

            try:
                exact_nll_float = float(exact_nll)
            except Exception:
                exact_nll_float = float("nan")

            try:
                exact_bits_float = float(exact_bits)
            except Exception:
                exact_bits_float = float("nan")

            exact_valid = (
                exact_candidate_count > 0
                and np.isfinite(exact_nll_float)
                and np.isfinite(exact_bits_float)
            )

            result["sparsepcgc_exact_teacher_valid"] = bool(exact_valid)
            result["sparsepcgc_exact_teacher_invalid_reason"] = ""

            if not exact_valid and bool(exact_occupancy):
                if exact_candidate_count <= 0:
                    result["sparsepcgc_exact_teacher_invalid_reason"] = "candidate_count_zero"
                elif not np.isfinite(exact_nll_float):
                    result["sparsepcgc_exact_teacher_invalid_reason"] = "nll_non_finite"
                elif not np.isfinite(exact_bits_float):
                    result["sparsepcgc_exact_teacher_invalid_reason"] = "bits_non_finite"
                else:
                    result["sparsepcgc_exact_teacher_invalid_reason"] = "unknown"
            node = float(result.get("node_count", result.get("node", 0.0)))
            single = float(result.get("single_child_count", result.get("single", 0.0)))
            stats = {
                "bit": bit,
                "bpp": bit / max(float(point_count), 1.0),
                "bpn": bit / max(float(node), 1.0),
                "single": single,
                "node": node,
                "point_count": point_count,
                "codec": self.codec_name,
                "mode": str(getattr(self.args, "sparsepcgc_mode", "dense_lossless")),
                "encode_time": float(time.time() - encode_t0),
                "sparsepcgc_input_prepare_time": float(input_prepare_time),
                "sparsepcgc_ply_write_time": float(ply_write_time + ascii_fallback_write_time),
                "sparsepcgc_worker_roundtrip_time": float(worker_roundtrip_time),
                "sparsepcgc_binary_ply_used": bool(used_binary_ply and not ascii_fallback_used),
                "sparsepcgc_ascii_ply_fallback_used": bool(ascii_fallback_used),
                "sparsepcgc_actual_result_cache_hit": False,
                "sparsepcgc_point_sha256": str(point_hash),
                "sparsepcgc_codec_fingerprint": self._codec_cache_fingerprint(),
            }
            for key, value in result.items():
                key_text = str(key)

                keep_key = (
                    key_text.startswith("sparsepcgc_")
                    or key_text.startswith("exact_")
                    or key_text.startswith("cuda_")
                    or key_text.startswith("gpu_")
                    or key_text.startswith("worker_cuda_")
                    or key_text.startswith("worker_gpu_")
                    or key_text.startswith("sparsepcgc_worker_cuda_")
                    or key_text.startswith("sparsepcgc_worker_gpu_")
                    or key_text
                    in {
                        "bitstream_path",
                        "decoded_path",
                        "decoded_copy_path",
                        "decoded_point_count",
                        "decoded_codec_point_count",
                    }
                )

                if keep_key:
                    stats[str(key)] = value
            # train.py側で扱いやすい短縮aliasも用意する。
            if "sparsepcgc_worker_cuda_allocated_mb" in stats:
                stats["actual_sparsepcgc_worker_cuda_allocated_mb"] = stats["sparsepcgc_worker_cuda_allocated_mb"]
            if "sparsepcgc_worker_cuda_reserved_mb" in stats:
                stats["actual_sparsepcgc_worker_cuda_reserved_mb"] = stats["sparsepcgc_worker_cuda_reserved_mb"]
            if "sparsepcgc_worker_cuda_max_allocated_mb" in stats:
                stats["actual_sparsepcgc_worker_cuda_max_allocated_mb"] = stats["sparsepcgc_worker_cuda_max_allocated_mb"]
            if "sparsepcgc_worker_cuda_max_reserved_mb" in stats:
                stats["actual_sparsepcgc_worker_cuda_max_reserved_mb"] = stats["sparsepcgc_worker_cuda_max_reserved_mb"]
            if "sparsepcgc_worker_cuda_allocated_delta_mb" in stats:
                stats["actual_sparsepcgc_worker_cuda_allocated_delta_mb"] = stats["sparsepcgc_worker_cuda_allocated_delta_mb"]
            if "sparsepcgc_worker_cuda_reserved_delta_mb" in stats:
                stats["actual_sparsepcgc_worker_cuda_reserved_delta_mb"] = stats["sparsepcgc_worker_cuda_reserved_delta_mb"]
            self._store_cached_result(result_cache_key, stats)
            if remove_workspace_after:
                shutil.rmtree(workspace, ignore_errors=True)
            return stats

    def encode_bits_many(self, candidates, *, max_parallel=1, mode="single", fallback_to_single=True):
        return _encode_bits_many_sequential(
            self,
            candidates,
            max_parallel=max_parallel,
            mode=mode,
            fallback_to_single=fallback_to_single,
        )

    def close(self):
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin is not None:
                    self._request_id += 1
                    proc.stdin.write(json.dumps({"command": "shutdown", "request_id": self._request_id}) + "\n")
                    proc.stdin.flush()
                proc.wait(timeout=5.0)
            except Exception:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except Exception:
                    proc.kill()
        if self._workspace_dir is not None:
            shutil.rmtree(self._workspace_dir, ignore_errors=True)
            self._workspace_dir = None
        self._result_cache.clear()
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _GPCCActualEncoder:
    """Thin loss-side wrapper around the myNet G-PCC geometry encoder."""

    codec_name = "gpcc"

    def __init__(self, args, writer=None):
        self.encoder = GPCCGeometryEncoder(
            root=getattr(args, "gpcc_root", ""),
            encoder_path=getattr(args, "gpcc_encoder_path", ""),
            cfg_dir=getattr(args, "gpcc_cfg_dir", ""),
            tmp_root=getattr(args, "gpcc_tmp_dir", ""),
            timeout=float(getattr(args, "gpcc_timeout", 120.0)),
            effective_qs=float(getattr(args, "gpcc_effective_qs", getattr(args, "qs", 1.0))),
            prequantize=bool(getattr(args, "gpcc_prequantize", True)),
            disable_attribute_coding=bool(getattr(args, "gpcc_disable_attribute_coding", True)),
            merge_duplicated_points=bool(getattr(args, "gpcc_merge_duplicated_points", True)),
            writer=writer,
        )

    def encode_bits(self, pts_3n):
        return self.encoder.encode_tensor(pts_3n)

    def encode_bits_many(self, candidates, *, max_parallel=1, mode="single", fallback_to_single=True):
        return _encode_bits_many_sequential(
            self,
            candidates,
            max_parallel=max_parallel,
            mode=mode,
            fallback_to_single=fallback_to_single,
        )


class _DracoActualEncoder:
    codec_name = "draco"

    def __init__(self, args, writer=None):
        self.encoder = DracoGeometryEncoder(
            root=getattr(args, "draco_root", ""),
            encoder_path=getattr(args, "draco_encoder_path", ""),
            decoder_path=getattr(args, "draco_decoder_path", ""),
            tmp_root=getattr(args, "draco_tmp_dir", ""),
            timeout=float(getattr(args, "draco_timeout", 120.0)),
            effective_qs=float(getattr(args, "draco_effective_qs", getattr(args, "qs", 1.0))),
            prequantize=bool(getattr(args, "draco_prequantize", True)),
            position_quantization_bits=int(getattr(args, "draco_position_quantization_bits", 0)),
            compression_level=int(getattr(args, "draco_compression_level", 7)),
            skip_decode=bool(getattr(args, "draco_skip_decode", True)),
            force_point_cloud=bool(getattr(args, "draco_force_point_cloud", True)),
            merge_duplicated_points=bool(getattr(args, "draco_merge_duplicated_points", True)),
            writer=writer,
        )

    def encode_bits(self, pts_3n):
        return self.encoder.encode_tensor(pts_3n)

    def encode_bits_many(self, candidates, *, max_parallel=1, mode="single", fallback_to_single=True):
        return _encode_bits_many_sequential(
            self,
            candidates,
            max_parallel=max_parallel,
            mode=mode,
            fallback_to_single=fallback_to_single,
        )


def _actual_codec_key(args):
    backend = str(getattr(args, "compression_loss_backend", "")).strip().lower()
    compress = str(getattr(args, "compress", "OctAttention")).strip().lower().replace("_", "").replace("-", "")
    if backend.startswith("sparsepcgc") or compress == "sparsepcgc":
        return "sparsepcgc"
    if backend.startswith("gpcc") or compress == "gpcc":
        return "gpcc"
    if backend.startswith("draco") or compress == "draco":
        return "draco"
    return "octattention"


def build_actual_encoder(args, writer=None):
    codec_key = _actual_codec_key(args)
    if codec_key == "sparsepcgc":
        return _SparsePCGCActualEncoder(args, writer=writer)
    if codec_key == "gpcc":
        return _GPCCActualEncoder(args, writer=writer)
    if codec_key == "draco":
        return _DracoActualEncoder(args, writer=writer)
    return _OctAttentionActualEncoder(args, writer=writer)


def _candidate_tensor_from_many_item(item):
    if torch.is_tensor(item):
        value = item
    elif isinstance(item, dict):
        value = None
        for key in ("pts", "xyz", "points", "candidate_xyz"):
            maybe_value = item.get(key, None)
            if torch.is_tensor(maybe_value):
                value = maybe_value
                break
    else:
        value = None
    if not torch.is_tensor(value):
        return None
    if value.ndim == 3 and value.shape[0] == 1 and value.shape[1] == 3:
        value = value.squeeze(0)
    if value.ndim == 2 and value.shape[0] == 3:
        return value
    if value.ndim == 2 and value.shape[1] == 3:
        return value.transpose(0, 1).contiguous()
    if value.ndim == 3 and value.shape[1] == 3:
        return value[0].contiguous()
    return value


def _encode_bits_many_sequential(encoder, candidates, *, max_parallel=1, mode="single", fallback_to_single=True):
    requested_mode = str(mode).strip().lower()
    effective_mode = "single"
    if requested_mode not in {"single", "worker_pool"}:
        requested_mode = "single"
    if requested_mode == "worker_pool" and int(max_parallel) > 1:
        if not bool(fallback_to_single):
            raise RuntimeError("worker_pool actual backend is not implemented in this patch.")
        effective_mode = "single_fallback"

    results = []
    for candidate_idx, candidate in enumerate(list(candidates or [])):
        pts_3n = _candidate_tensor_from_many_item(candidate)
        base_result = {
            "candidate_index": int(candidate_idx),
            "actual_requested": pts_3n is not None,
            "actual_finished": False,
            "actual_worker_id": -1,
            "actual_wall_time": 0.0,
            "actual_error_reason": "",
            "actual_parallel_mode_effective": effective_mode,
        }
        if isinstance(candidate, dict):
            for key in ("candidate_id", "candidate_class", "candidate_source"):
                if key in candidate:
                    base_result[key] = candidate.get(key)

        if pts_3n is None:
            base_result["actual_error_reason"] = "candidate_tensor_missing"
            results.append(base_result)
            continue

        wall_t0 = time.time()
        try:
            stats = dict(encoder.encode_bits(pts_3n))
            base_result.update(stats)
            base_result["actual_finished"] = True
        except Exception as exc:
            base_result["actual_error_reason"] = str(exc)
        base_result["actual_wall_time"] = float(time.time() - wall_t0)
        results.append(base_result)
    return results
