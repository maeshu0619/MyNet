"""encoder freeze後だけ使用できるLayer B Student Feature Cache。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, Mapping

import numpy as np
import torch

from .exact_teacher_cache import _exclusive_lock, sha256_file


SCHEMA_VERSION = "mynet_frozen_student_feature_cache_v1"


def feature_fingerprint(
    *, layer_a_key: str, encoder_checkpoint: str, encoder_source: str,
    dtype: str, receptive_field: int, tile_partition: Mapping[str, object],
) -> Dict[str, object]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "layer_a_key": str(layer_a_key),
        "encoder_checkpoint_path": str(Path(encoder_checkpoint).resolve()),
        "encoder_checkpoint_sha256": sha256_file(encoder_checkpoint),
        "encoder_source_path": str(Path(encoder_source).resolve()),
        "encoder_source_sha256": sha256_file(encoder_source),
        "dtype": str(dtype),
        "receptive_field": int(receptive_field),
        "tile_partition": dict(tile_partition),
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    value["fingerprint_sha256"] = hashlib.sha256(encoded).hexdigest()
    return value


def _array_checksum(metadata: Mapping[str, object], arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(
        json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


class FrozenStudentFeatureCache:
    def __init__(self, root: str):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, fingerprint: Mapping[str, object]) -> Path:
        return self.root / ("student_features_" + str(fingerprint["fingerprint_sha256"]) + ".npz")

    def write(self, fingerprint: Mapping[str, object], tensors: Mapping[str, torch.Tensor]) -> Path:
        path = self.path_for(fingerprint)
        arrays = {
            name: value.detach().cpu().contiguous().numpy()
            for name, value in tensors.items()
        }
        metadata = {"fingerprint": dict(fingerprint), "complete": True}
        checksum = _array_checksum(metadata, arrays)
        lock = path.with_suffix(path.suffix + ".lock")
        with _exclusive_lock(lock):
            if path.is_file():
                self.load(fingerprint)
                return path
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                with temporary.open("wb") as stream:
                    np.savez_compressed(
                        stream,
                        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                        payload_sha256=np.asarray(checksum),
                        **arrays,
                    )
                os.replace(str(temporary), str(path))
            finally:
                if temporary.exists():
                    temporary.unlink()
        return path

    def load(self, fingerprint: Mapping[str, object]) -> Dict[str, torch.Tensor]:
        path = self.path_for(fingerprint)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            with np.load(str(path), allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"].item()))
                checksum = str(payload["payload_sha256"].item())
                arrays = {
                    name: np.ascontiguousarray(payload[name])
                    for name in payload.files
                    if name not in {"metadata_json", "payload_sha256"}
                }
        except Exception as error:
            raise RuntimeError("Frozen Student Feature Cacheが破損している") from error
        if dict(metadata.get("fingerprint") or {}) != dict(fingerprint):
            raise RuntimeError("encoder更新またはtile条件変更によりLayer Bを無効化した")
        if not metadata.get("complete") or checksum != _array_checksum(metadata, arrays):
            raise RuntimeError("Frozen Student Feature Cache checksum不一致")
        return {name: torch.from_numpy(value.copy()) for name, value in arrays.items()}

