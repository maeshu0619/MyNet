# dataset.py
import os
from pathlib import Path
import numpy as np
import torch

_PLY_CACHE = {}
_MASTER_ROOT = Path(__file__).resolve().parents[4]
_DATA_ROOT = (_MASTER_ROOT / "../../../data/maejima/data").resolve()


def _resolve_data_path(path):
    raw = Path(path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((Path.cwd() / raw).resolve())
        candidates.append((_MASTER_ROOT / raw).resolve())

    normalized = str(raw).replace("\\", "/")
    marker = "data/"
    if marker in normalized:
        tail = normalized.split(marker, 1)[1].lstrip("/")
        candidates.append((_DATA_ROOT / tail).resolve())

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0] if candidates else raw)

# def load_ply(path):
#     pcd = o3d.io.read_point_cloud(path)
#     points = np.asarray(pcd.points, dtype=np.float32) 
#     return points

def _numpy_dtype_from_ply_type(fmt, prop_type):
    endian = "<" if fmt == "binary_little_endian" else ">"
    mapping = {
        "char": "i1",
        "uchar": "u1",
        "int8": "i1",
        "uint8": "u1",
        "short": "i2",
        "ushort": "u2",
        "int16": "i2",
        "uint16": "u2",
        "int": "i4",
        "uint": "u4",
        "int32": "i4",
        "uint32": "u4",
        "float": "f4",
        "float32": "f4",
        "double": "f8",
        "float64": "f8",
    }
    code = mapping[prop_type]
    if code in {"i1", "u1"}:
        return code
    return endian + code


def _load_ply_numpy(path, return_color=True):
    with open(path, "rb") as f:
        fmt = None
        vertex_count = None
        properties = []
        in_vertex = False
        while True:
            line_b = f.readline()
            if not line_b:
                raise ValueError(f"Invalid PLY header: {path}")
            line = line_b.decode("ascii", errors="replace").strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element "):
                parts = line.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and line.startswith("property "):
                parts = line.split()
                if len(parts) == 3:
                    properties.append((parts[2], parts[1]))
            elif line == "end_header":
                break

        if fmt is None or vertex_count is None:
            raise ValueError(f"PLY header misses format or vertex count: {path}")
        if fmt in {"binary_little_endian", "binary_big_endian"}:
            dtype = np.dtype([(name, _numpy_dtype_from_ply_type(fmt, typ)) for name, typ in properties])
            data = np.fromfile(f, dtype=dtype, count=vertex_count)
            xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
            if not return_color:
                return xyz
            if all(name in data.dtype.names for name in ("red", "green", "blue")):
                rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.float32) / 255.0
            else:
                rgb = np.zeros_like(xyz, dtype=np.float32)
            return np.concatenate([xyz, rgb], axis=1).astype(np.float32)
        if fmt == "ascii":
            rows = []
            for _ in range(vertex_count):
                rows.append(f.readline().decode("ascii", errors="replace").split())
            arr = np.asarray(rows, dtype=np.float32)
            names = [name for name, _ in properties]
            xyz = np.stack([arr[:, names.index("x")], arr[:, names.index("y")], arr[:, names.index("z")]], axis=1).astype(np.float32)
            if not return_color:
                return xyz
            if all(name in names for name in ("red", "green", "blue")):
                rgb = np.stack([arr[:, names.index("red")], arr[:, names.index("green")], arr[:, names.index("blue")]], axis=1).astype(np.float32) / 255.0
            else:
                rgb = np.zeros_like(xyz, dtype=np.float32)
            return np.concatenate([xyz, rgb], axis=1).astype(np.float32)
        raise ValueError(f"Unsupported PLY format {fmt}: {path}")


def _try_import_open3d():
    try:
        import open3d as _o3d
    except Exception:
        return None
    return _o3d


def _validate_loaded_points(points, path):
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        raise ValueError(f"Empty or invalid point cloud: {path} (shape={points.shape})")
    return np.ascontiguousarray(points, dtype=np.float32)


def _load_ply_open3d(path, return_color=True):
    o3d = _try_import_open3d()
    if o3d is None:
        return _load_ply_numpy(path, return_color=return_color)

    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points, dtype=np.float32)

    if not return_color:
        return xyz

    if pcd.has_colors():
        rgb = np.asarray(pcd.colors, dtype=np.float32)
    else:
        rgb = np.zeros_like(xyz, dtype=np.float32)

    return np.concatenate([xyz, rgb], axis=1).astype(np.float32)


def load_ply(path, return_color=True, loader="numpy"):
    loader = str(loader or "numpy").strip().lower()
    if loader in {"numpy", "np"}:
        return _validate_loaded_points(_load_ply_numpy(path, return_color=return_color), path)
    if loader in {"open3d", "o3d"}:
        return _validate_loaded_points(_load_ply_open3d(path, return_color=return_color), path)
    if loader == "auto":
        try:
            points = _load_ply_numpy(path, return_color=return_color)
        except Exception:
            points = _load_ply_open3d(path, return_color=return_color)
        return _validate_loaded_points(points, path)
    raise ValueError(f"Unsupported PLY loader: {loader}")


def clear_ply_cache():
    _PLY_CACHE.clear()


class PlyDirDataset(torch.utils.data.Dataset):
    """
    - ディレクトリが渡された場合：
        その直下の .ply をすべて扱う
    - 単一 .ply ファイルが渡された場合：
        そのファイルのみを扱う
    """
    def __init__(self, args, path):
        path = _resolve_data_path(path)
        self.use_cache = bool(getattr(args, "dataset_cache", True))
        self.ply_loader = str(getattr(args, "ply_loader", "numpy")).strip().lower()
        if args.trainORtest == "train":
            max_files = args.max_files
        else:
            max_files = args.max_files_test
        # 単一 ply ファイルの場合
        if os.path.isfile(path):
            if not path.endswith(".ply"):
                raise ValueError(f"Input file is not a .ply file: {path}")
            self.files = [path]

        # ディレクトリの場合
        elif os.path.isdir(path):
            self.files = [
                os.path.join(path, f)
                for f in sorted(os.listdir(path))
                if f.endswith(".ply")
            ]
            if len(self.files) == 0:
                raise ValueError(f"No .ply files found in directory: {path}")
            self.files = self.files[:max_files]

        else:
            raise ValueError(f"Invalid path: {path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        if self.use_cache:
            cached = _PLY_CACHE.get(path)
            if cached is not None:
                return cached

        points = load_ply(path, loader=self.ply_loader)
        points = torch.from_numpy(points)
        if self.use_cache:
            _PLY_CACHE[path] = points
        return points

def collect_seq_dirs(root):
    root = _resolve_data_path(root)
    seq_dirs = []
    with os.scandir(root) as root_entries:
        for dataset_entry in root_entries:
            if not dataset_entry.is_dir():
                continue
            with os.scandir(dataset_entry.path) as seq_entries:
                for seq_entry in seq_entries:
                    if seq_entry.is_dir():
                        seq_dirs.append(seq_entry.path)
    return sorted(seq_dirs)

def collect_seq_dirs2(root, dataset_name=None):
    """
    root:
        ../data/train/video_noised など
    dataset_name:
        "UVG" や "CWI" を指定
        None の場合は従来どおり全datasetを対象
    """
    root = _resolve_data_path(root)
    seq_dirs = []

    if dataset_name is not None:
        # 指定された dataset のみ
        d1 = os.path.join(root, dataset_name)
        if not os.path.isdir(d1):
            raise ValueError(f"Dataset not found: {d1}")

        with os.scandir(d1) as seq_entries:
            for seq_entry in sorted(seq_entries, key=lambda entry: entry.name):
                if seq_entry.is_dir():
                    seq_dirs.append(seq_entry.path)

    else:
        # 従来どおり全 dataset
        with os.scandir(root) as root_entries:
            for dataset_entry in sorted(root_entries, key=lambda entry: entry.name):
                if not dataset_entry.is_dir():
                    continue
                with os.scandir(dataset_entry.path) as seq_entries:
                    for seq_entry in sorted(seq_entries, key=lambda entry: entry.name):
                        if seq_entry.is_dir():
                            seq_dirs.append(seq_entry.path)

    return seq_dirs
