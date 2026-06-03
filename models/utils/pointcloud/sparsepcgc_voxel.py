from typing import Any, Dict, Optional, Tuple

import torch

def quantize_sparsepcgc_coords(pts_xyz, args, coord_scale=None, offset=None, return_metadata=False):
    """
    Phase1/Phase2互換用の量子化入口である。
    既存の canonical_sparsepcgc_voxel_coords を正式実装として使う。
    """
    return canonical_sparsepcgc_voxel_coords(
        pts_xyz,
        args=args,
        coord_scale=coord_scale,
        global_offset=offset,
        return_metadata=return_metadata,
    )


def dequantize_sparsepcgc_coords(
    coords,
    meta=None,
    args=None,
    center=False,
    dtype=None,
    device=None,
):
    """
    canonical voxel coordsをxyzへ戻す。
    center=True の場合は、voxel中心へ半stepだけ寄せる。
    既定は center=False とし、既存の global_offset + coords * global_qs を壊さない。
    """
    coords_b3n = normalize_voxel_coords_b3n(coords, device=device)
    out_dtype = dtype if dtype is not None else torch.float32

    if meta is None:
        meta = resolve_sparsepcgc_quant_metadata(
            coords_b3n,
            args=args,
            coord_scale=None,
            global_offset=None,
        )

    xyz = sparsepcgc_voxel_coords_to_xyz(
        coords_b3n,
        metadata=meta,
        args=args,
        dtype=out_dtype,
    )

    if bool(center):
        if isinstance(meta, dict) and "effective_qs_tensor" in meta:
            step = meta["effective_qs_tensor"].to(device=xyz.device, dtype=xyz.dtype)
        elif isinstance(meta, dict) and "global_qs" in meta:
            step_raw = meta["global_qs"]
            if torch.is_tensor(step_raw):
                step = step_raw.to(device=xyz.device, dtype=xyz.dtype).reshape(-1, 1, 1)
                if step.shape[0] == 1 and xyz.shape[0] > 1:
                    step = step.expand(xyz.shape[0], -1, -1)
            else:
                step = xyz.new_full((xyz.shape[0], 1, 1), float(step_raw))
        else:
            step = xyz.new_full((xyz.shape[0], 1, 1), sparsepcgc_effective_qs_value(args))
        xyz = xyz + 0.5 * step.clamp_min(1e-12)

    return xyz


def attach_sparsepcgc_voxel_meta(octree_context, coords, meta):
    """
    octree_contextへcanonical voxel情報を付与する。
    既存のdictを破壊しないため、新しいdictを返す。
    """
    out = dict(octree_context or {})
    out["global_voxel_coords"] = coords
    if isinstance(meta, dict):
        if "global_qs" in meta:
            out["global_qs"] = meta["global_qs"]
        elif "effective_qs" in meta:
            out["global_qs"] = meta["effective_qs"]

        if "global_offset_tensor" in meta:
            out["global_offset"] = meta["global_offset_tensor"]
        elif "global_offset" in meta:
            out["global_offset"] = meta["global_offset"]

        out["sparsepcgc_voxel_meta"] = meta
    return out


def normalize_voxel_coords_b3n(coords, device=None):
    """
    voxel coordsを[B, 3, N]のlong tensorに正規化する。
    入力として[3,N], [N,3], [B,3,N], [B,N,3]を受け付ける。
    """
    if coords is None:
        raise ValueError("coords must not be None.")

    if not torch.is_tensor(coords):
        coords = torch.as_tensor(coords)

    if device is not None:
        coords = coords.to(device=device)

    if coords.ndim == 2:
        if coords.shape[0] == 3:
            coords = coords.unsqueeze(0)
        elif coords.shape[1] == 3:
            coords = coords.transpose(0, 1).contiguous().unsqueeze(0)
        else:
            raise ValueError(f"coords must be [3,N] or [N,3], got shape={tuple(coords.shape)}")
    elif coords.ndim == 3:
        if coords.shape[1] == 3:
            coords = coords.contiguous()
        elif coords.shape[2] == 3:
            coords = coords.permute(0, 2, 1).contiguous()
        else:
            raise ValueError(f"coords must be [B,3,N] or [B,N,3], got shape={tuple(coords.shape)}")
    else:
        raise ValueError(f"coords must be 2D or 3D, got shape={tuple(coords.shape)}")

    return coords.to(dtype=torch.long).contiguous()


def unique_voxel_coords_batched(coords):
    """
    各batchごとに重複voxel coordsを除去する。
    戻り値はpadding済みcoords、valid_mask、各batchのunique数を含むdictである。
    """
    coords_b3n = normalize_voxel_coords_b3n(coords)
    B, _, _ = coords_b3n.shape

    unique_list = []
    counts = []
    for b in range(B):
        coords_n3 = coords_b3n[b].transpose(0, 1).contiguous()
        if coords_n3.numel() == 0:
            unique_b = coords_n3.new_empty((0, 3))
        else:
            unique_b = torch.unique(coords_n3, dim=0, sorted=True)
        unique_list.append(unique_b)
        counts.append(int(unique_b.shape[0]))

    max_count = max(counts) if counts else 0
    if max_count <= 0:
        unique_coords = coords_b3n.new_empty((B, 3, 0))
        valid_mask = torch.zeros((B, 0), device=coords_b3n.device, dtype=torch.bool)
        count_tensor = torch.zeros((B,), device=coords_b3n.device, dtype=torch.long)
        return {
            "coords": unique_coords,
            "valid_mask": valid_mask,
            "counts": count_tensor,
        }

    padded = coords_b3n.new_zeros((B, 3, max_count))
    valid_mask = torch.zeros((B, max_count), device=coords_b3n.device, dtype=torch.bool)

    for b, unique_b in enumerate(unique_list):
        count = int(unique_b.shape[0])
        if count <= 0:
            continue
        padded[b, :, :count] = unique_b.transpose(0, 1).contiguous()
        valid_mask[b, :count] = True

    count_tensor = torch.as_tensor(counts, device=coords_b3n.device, dtype=torch.long)
    return {
        "coords": padded,
        "valid_mask": valid_mask,
        "counts": count_tensor,
    }


def restore_points_from_voxel_coords(
    coords,
    meta=None,
    args=None,
    center=None,
    unique=True,
    dtype=None,
    device=None,
):
    """
    canonical voxel coordsから点群xyzへ戻す。
    unique=Trueなら同一voxelを1点にまとめる。
    戻り値は xyz, restore_info である。
    """
    coords_b3n = normalize_voxel_coords_b3n(coords, device=device)
    input_points = int(coords_b3n.shape[-1])

    if center is None:
        center = bool(getattr(args, "sparsepcgc_dequantize_center", False))

    if bool(unique):
        unique_result = unique_voxel_coords_batched(coords_b3n)
        restore_coords = unique_result["coords"]
        valid_mask = unique_result["valid_mask"]
        counts = unique_result["counts"]
    else:
        restore_coords = coords_b3n
        valid_mask = torch.ones(
            (coords_b3n.shape[0], coords_b3n.shape[-1]),
            device=coords_b3n.device,
            dtype=torch.bool,
        )
        counts = torch.full(
            (coords_b3n.shape[0],),
            int(coords_b3n.shape[-1]),
            device=coords_b3n.device,
            dtype=torch.long,
        )

    xyz = dequantize_sparsepcgc_coords(
        restore_coords,
        meta=meta,
        args=args,
        center=bool(center),
        dtype=dtype,
        device=device,
    )

    restore_info = {
        "restore_input_points": int(input_points),
        "restore_output_points": int(restore_coords.shape[-1]),
        "restore_counts": counts.detach(),
        "restore_valid_mask": valid_mask.detach(),
        "restore_unique": bool(unique),
        "restore_center": bool(center),
        "restore_has_meta": bool(isinstance(meta, dict)),
    }
    return xyz, restore_info

def normalize_compress_key(raw_value: Any) -> str:
    return (
        str(raw_value)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def is_sparsepcgc_args(args: Any) -> bool:
    return normalize_compress_key(getattr(args, "compress", "")) == "sparsepcgc"


def sparsepcgc_voxel_size_value(args: Any = None, default: float = 1.0) -> float:
    if args is None:
        return max(float(default), 1e-12)
    return max(float(getattr(args, "sparsepcgc_voxel_size", default)), 1e-12)


def sparsepcgc_pos_quantscale_value(args: Any = None, default: int = 1) -> int:
    if args is None:
        return max(int(default), 1)
    return max(int(getattr(args, "sparsepcgc_pos_quantscale", default)), 1)


def sparsepcgc_effective_qs_value(args: Any = None, default_voxel_size: float = 1.0, default_pos_q: int = 1) -> float:
    voxel_size = sparsepcgc_voxel_size_value(args, default=default_voxel_size)
    pos_q = sparsepcgc_pos_quantscale_value(args, default=default_pos_q)

    if args is not None:
        raw_effective = float(getattr(args, "sparsepcgc_effective_qs", 0.0))
        if raw_effective > 0.0:
            return max(raw_effective, 1e-12)

    return max(float(voxel_size) * float(pos_q), 1e-12)


def _as_b3n(pts_xyz: torch.Tensor) -> Tuple[torch.Tensor, str]:
    if pts_xyz is None:
        raise ValueError("pts_xyz must not be None.")

    if not torch.is_tensor(pts_xyz):
        pts_xyz = torch.as_tensor(pts_xyz)

    if pts_xyz.ndim == 2:
        if pts_xyz.shape[0] == 3:
            return pts_xyz.unsqueeze(0), "3n"
        if pts_xyz.shape[1] == 3:
            return pts_xyz.transpose(0, 1).contiguous().unsqueeze(0), "n3"
        raise ValueError(f"pts_xyz must be [3,N] or [N,3], got shape={tuple(pts_xyz.shape)}")

    if pts_xyz.ndim == 3:
        if pts_xyz.shape[1] == 3:
            return pts_xyz, "b3n"
        if pts_xyz.shape[2] == 3:
            return pts_xyz.permute(0, 2, 1).contiguous(), "bn3"
        raise ValueError(f"pts_xyz must be [B,3,N] or [B,N,3], got shape={tuple(pts_xyz.shape)}")

    raise ValueError(f"pts_xyz must be 2D or 3D, got shape={tuple(pts_xyz.shape)}")


def _restore_layout(tensor_b3n: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "3n":
        return tensor_b3n.squeeze(0)
    if layout == "n3":
        return tensor_b3n.squeeze(0).transpose(0, 1).contiguous()
    if layout == "b3n":
        return tensor_b3n
    if layout == "bn3":
        return tensor_b3n.permute(0, 2, 1).contiguous()
    raise ValueError(f"unknown layout: {layout}")


def _coord_scale_b11(pts_b3n: torch.Tensor, coord_scale: Any = None) -> torch.Tensor:
    if coord_scale is None:
        return pts_b3n.new_ones((pts_b3n.shape[0], 1, 1))

    if torch.is_tensor(coord_scale):
        scale = coord_scale.to(device=pts_b3n.device, dtype=pts_b3n.dtype).reshape(-1, 1, 1)
        if scale.shape[0] == 1 and pts_b3n.shape[0] > 1:
            scale = scale.expand(pts_b3n.shape[0], -1, -1)
        if scale.shape[0] != pts_b3n.shape[0]:
            raise ValueError(
                f"coord_scale batch size mismatch: coord_scale={tuple(scale.shape)}, pts={tuple(pts_b3n.shape)}"
            )
        return scale.clamp_min(1e-12)

    return pts_b3n.new_full((pts_b3n.shape[0], 1, 1), max(float(coord_scale), 1e-12))


def _offset_b3n(pts_b3n: torch.Tensor, global_offset: Any = None) -> torch.Tensor:
    if global_offset is None:
        return pts_b3n.new_zeros((pts_b3n.shape[0], 3, 1))

    if torch.is_tensor(global_offset):
        offset = global_offset.to(device=pts_b3n.device, dtype=pts_b3n.dtype)
    else:
        offset = torch.as_tensor(global_offset, device=pts_b3n.device, dtype=pts_b3n.dtype)

    if offset.ndim == 1 and offset.numel() == 3:
        offset = offset.view(1, 3, 1)
    elif offset.ndim == 2 and offset.shape[-1] == 3:
        offset = offset.view(-1, 3, 1)
    elif offset.ndim == 2 and offset.shape[0] == 3:
        offset = offset.unsqueeze(0)
    elif offset.ndim == 3 and offset.shape[1] == 3:
        offset = offset[:, :, :1]
    else:
        raise ValueError(f"global_offset must be [3], [B,3], [3,1], or [B,3,1], got shape={tuple(offset.shape)}")

    if offset.shape[0] == 1 and pts_b3n.shape[0] > 1:
        offset = offset.expand(pts_b3n.shape[0], -1, -1)

    if offset.shape[0] != pts_b3n.shape[0]:
        raise ValueError(
            f"global_offset batch size mismatch: offset={tuple(offset.shape)}, pts={tuple(pts_b3n.shape)}"
        )

    return offset.contiguous()


def sparsepcgc_voxel_size_tensor(
    pts_xyz: torch.Tensor,
    args: Any = None,
    coord_scale: Any = None,
    *,
    voxel_size: Optional[float] = None,
    voxel_scale: float = 1.0,
) -> torch.Tensor:
    pts_b3n, _ = _as_b3n(pts_xyz)
    base_voxel = sparsepcgc_voxel_size_value(args, default=1.0) if voxel_size is None else max(float(voxel_size), 1e-12)
    scale = _coord_scale_b11(pts_b3n, coord_scale)
    return pts_b3n.new_full((pts_b3n.shape[0], 1, 1), float(base_voxel) * float(voxel_scale)) / scale


def sparsepcgc_effective_qs_tensor(
    pts_xyz: torch.Tensor,
    args: Any = None,
    coord_scale: Any = None,
    *,
    voxel_size: Optional[float] = None,
    pos_quantscale: Optional[int] = None,
    voxel_scale: float = 1.0,
) -> torch.Tensor:
    voxel = sparsepcgc_voxel_size_tensor(
        pts_xyz,
        args=args,
        coord_scale=coord_scale,
        voxel_size=voxel_size,
        voxel_scale=voxel_scale,
    )
    pos_q = sparsepcgc_pos_quantscale_value(args, default=1) if pos_quantscale is None else max(int(pos_quantscale), 1)
    return voxel * float(pos_q)


def resolve_sparsepcgc_quant_metadata(
    pts_xyz: Optional[torch.Tensor] = None,
    args: Any = None,
    coord_scale: Any = None,
    *,
    voxel_size: Optional[float] = None,
    pos_quantscale: Optional[int] = None,
    voxel_scale: float = 1.0,
    global_offset: Any = None,
    quant_mode: Optional[str] = None,
) -> Dict[str, Any]:
    base_voxel = sparsepcgc_voxel_size_value(args, default=1.0) if voxel_size is None else max(float(voxel_size), 1e-12)
    pos_q = sparsepcgc_pos_quantscale_value(args, default=1) if pos_quantscale is None else max(int(pos_quantscale), 1)
    effective_qs = max(float(base_voxel) * float(pos_q), 1e-12)

    mode = str(
        quant_mode
        if quant_mode is not None
        else getattr(args, "sparsepcgc_quant_mode", "round_voxel_then_pos")
    ).strip().lower()

    metadata: Dict[str, Any] = {
        "voxel_size": float(base_voxel),
        "pos_quantscale": int(pos_q),
        "effective_qs": float(effective_qs),
        "global_qs": float(effective_qs),
        "quant_mode": mode,
        "canonical_coord_format": "b3n_long",
    }

    if pts_xyz is not None:
        pts_b3n, _ = _as_b3n(pts_xyz)
        metadata["voxel_size_tensor"] = sparsepcgc_voxel_size_tensor(
            pts_b3n,
            args=args,
            coord_scale=coord_scale,
            voxel_size=base_voxel,
            voxel_scale=voxel_scale,
        )
        metadata["effective_qs_tensor"] = metadata["voxel_size_tensor"] * float(pos_q)
        metadata["global_offset_tensor"] = _offset_b3n(pts_b3n, global_offset)

    return metadata


def canonical_sparsepcgc_voxel_coords(
    pts_xyz: torch.Tensor,
    args: Any = None,
    coord_scale: Any = None,
    *,
    voxel_size: Optional[float] = None,
    pos_quantscale: Optional[int] = None,
    voxel_scale: float = 1.0,
    global_offset: Any = None,
    return_metadata: bool = False,
):
    pts_b3n, layout = _as_b3n(pts_xyz)
    metadata = resolve_sparsepcgc_quant_metadata(
        pts_b3n,
        args=args,
        coord_scale=coord_scale,
        voxel_size=voxel_size,
        pos_quantscale=pos_quantscale,
        voxel_scale=voxel_scale,
        global_offset=global_offset,
    )

    voxel = metadata["voxel_size_tensor"].to(device=pts_b3n.device, dtype=pts_b3n.dtype).clamp_min(1e-12)
    offset = metadata["global_offset_tensor"].to(device=pts_b3n.device, dtype=pts_b3n.dtype)
    pos_q = float(metadata["pos_quantscale"])

    # SparsePCGC互換の2段階量子化である。
    # 既存実装の round(pts / voxel_size) → round(coords / posQuantscale) と同じ意味にする。
    coords = torch.round((pts_b3n - offset) / voxel)
    if pos_q > 1.0:
        coords = torch.round(coords / pos_q)

    coords = coords.to(torch.long).contiguous()
    coords = _restore_layout(coords, layout)

    if return_metadata:
        return coords, metadata
    return coords


def sparsepcgc_voxel_coords_to_xyz(
    voxel_coords: torch.Tensor,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    args: Any = None,
    coord_scale: Any = None,
    voxel_size: Optional[float] = None,
    pos_quantscale: Optional[int] = None,
    voxel_scale: float = 1.0,
    global_offset: Any = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    coords_b3n, layout = _as_b3n(voxel_coords)
    out_dtype = dtype if dtype is not None else torch.float32

    coords_b3n = coords_b3n.to(dtype=out_dtype)

    if metadata is None:
        metadata = resolve_sparsepcgc_quant_metadata(
            coords_b3n,
            args=args,
            coord_scale=coord_scale,
            voxel_size=voxel_size,
            pos_quantscale=pos_quantscale,
            voxel_scale=voxel_scale,
            global_offset=global_offset,
        )

    if "effective_qs_tensor" in metadata:
        step = metadata["effective_qs_tensor"].to(device=coords_b3n.device, dtype=out_dtype)
    else:
        base_voxel = float(metadata.get("voxel_size", sparsepcgc_voxel_size_value(args, 1.0)))
        pos_q = int(metadata.get("pos_quantscale", sparsepcgc_pos_quantscale_value(args, 1)))
        step = coords_b3n.new_full((coords_b3n.shape[0], 1, 1), base_voxel * float(pos_q))

    if "global_offset_tensor" in metadata:
        offset = metadata["global_offset_tensor"].to(device=coords_b3n.device, dtype=out_dtype)
    else:
        offset = _offset_b3n(coords_b3n, global_offset).to(dtype=out_dtype)

    xyz = offset + coords_b3n * step.clamp_min(1e-12)
    return _restore_layout(xyz, layout)