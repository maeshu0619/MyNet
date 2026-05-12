from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


PLY_COORDINATES = {"x", "y", "z"}
PLY_TYPE_FORMATS = {
    "char": ("i1", 1),
    "int8": ("i1", 1),
    "uchar": ("u1", 1),
    "uint8": ("u1", 1),
    "short": ("<i2", 2),
    "int16": ("<i2", 2),
    "ushort": ("<u2", 2),
    "uint16": ("<u2", 2),
    "int": ("<i4", 4),
    "int32": ("<i4", 4),
    "uint": ("<u4", 4),
    "uint32": ("<u4", 4),
    "float": ("<f4", 4),
    "float32": ("<f4", 4),
    "double": ("<f8", 8),
    "float64": ("<f8", 8),
}


def parse_ply_header(path: str | Path) -> Dict[str, object]:
    path = Path(path)
    header_lines: List[str] = []
    with path.open("rb") as file:
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            decoded = line.decode("ascii", errors="replace").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                data_offset = file.tell()
                break

    properties: List[Dict[str, object]] = []
    result: Dict[str, object] = {
        "format": "",
        "point_count": 0,
        "properties": properties,
        "header_line_count": len(header_lines),
        "data_offset": data_offset,
    }
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "format":
            result["format"] = parts[1]
        elif len(parts) >= 3 and parts[:2] == ["element", "vertex"]:
            result["point_count"] = int(parts[2])
            in_vertex = True
        elif len(parts) >= 2 and parts[0] == "element":
            in_vertex = False
        elif in_vertex and len(parts) >= 3 and parts[0] == "property" and parts[1] != "list":
            properties.append({"type": parts[1], "name": parts[2]})
    return result


def read_ply_xyz(path: str | Path) -> np.ndarray:
    path = Path(path)
    parsed = parse_ply_header(path)
    point_count = int(parsed["point_count"])
    properties = list(parsed["properties"])
    coord_indices = [idx for idx, item in enumerate(properties) if item["name"] in PLY_COORDINATES]
    if len(coord_indices) < 3:
        raise ValueError(f"PLY does not contain x/y/z vertex properties: {path}")

    if parsed["format"] == "ascii":
        data = np.loadtxt(str(path), skiprows=int(parsed["header_line_count"]), max_rows=point_count)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data[:, coord_indices[:3]].astype(np.float64, copy=False)

    if parsed["format"] != "binary_little_endian":
        raise ValueError(f"Unsupported PLY format: {parsed['format']} ({path})")

    dtype_fields: List[Tuple[str, str]] = []
    for index, item in enumerate(properties):
        dtype = PLY_TYPE_FORMATS.get(str(item["type"]))
        if dtype is None:
            raise ValueError(f"Unsupported PLY property type {item['type']} in {path}")
        dtype_fields.append((f"f{index}", dtype[0]))
    dtype = np.dtype(dtype_fields)
    with path.open("rb") as file:
        file.seek(int(parsed["data_offset"]))
        data = np.fromfile(file, dtype=dtype, count=point_count)
    coords = np.column_stack([data[f"f{idx}"] for idx in coord_indices[:3]])
    return coords.astype(np.float64, copy=False)


def write_ascii_ply_xyz(path: str | Path, points: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with path.open("w", encoding="ascii") as file:
        file.write(header)
        np.savetxt(file, points, fmt="%.10g %.10g %.10g")


def point_count(path: str | Path) -> int:
    return int(parse_ply_header(path)["point_count"])
