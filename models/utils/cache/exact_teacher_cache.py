"""訓練専用Exact Teacher Cacheの検証済みI/O。

このcacheは教師ラベルだけを保持し、Network forwardや推論から参照しない。
保存済みActualを補間せず、旧schemaの欠損値も欠損maskのまま維持する。
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Mapping


SCHEMA_VERSION = "mynet_exact_teacher_cache_v1"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def payload_checksum(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("payload_sha256", None)
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def build_fingerprint(
    *,
    input_path: str,
    codec: Mapping[str, Any],
    source_files: Iterable[str],
    geometry: Mapping[str, Any],
) -> Dict[str, Any]:
    """入力・codec・Teacher実装を全て含む再利用fingerprintを返す。"""
    resolved = str(Path(input_path).expanduser().resolve())
    sources = []
    for raw in source_files:
        path = str(Path(raw).expanduser().resolve())
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        sources.append({"path": path, "sha256": sha256_file(path)})
    value = {
        "input_path": resolved,
        "input_sha256": sha256_file(resolved),
        "codec": dict(codec),
        "geometry": dict(geometry),
        "teacher_sources": sources,
    }
    value["fingerprint_sha256"] = hashlib.sha256(_canonical_bytes(value)).hexdigest()
    return value


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class ExactTeacherCache:
    """checksum/fingerprintを検証してatomicに保存する訓練専用cache。"""

    def __init__(self, root: str):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, fingerprint_sha256: str) -> Path:
        key = str(fingerprint_sha256).strip().lower()
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("fingerprint_sha256が不正である")
        return self.root / ("exact_teacher_" + key + ".json.gz")

    def load(self, fingerprint: Mapping[str, Any]) -> Dict[str, Any]:
        expected = str(fingerprint.get("fingerprint_sha256", ""))
        path = self.path_for(expected)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            with gzip.open(str(path), "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
        except Exception as error:
            raise RuntimeError("Exact Teacher Cacheが破損している: {}".format(path)) from error
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("Exact Teacher Cache schemaが一致しない")
        if dict(payload.get("fingerprint") or {}) != dict(fingerprint):
            raise RuntimeError("Exact Teacher Cache fingerprintが一致しない")
        if str(payload.get("payload_sha256", "")) != payload_checksum(payload):
            raise RuntimeError("Exact Teacher Cache checksumが一致しない")
        if not bool(payload.get("complete", False)):
            raise RuntimeError("partial Exact Teacher Cacheを拒否した")
        if not bool(payload.get("training_only", False)):
            raise RuntimeError("推論利用を許すTeacher Cacheは拒否する")
        return payload

    def write(self, fingerprint: Mapping[str, Any], content: Mapping[str, Any]) -> Path:
        expected = str(fingerprint.get("fingerprint_sha256", ""))
        path = self.path_for(expected)
        lock_path = path.with_suffix(path.suffix + ".lock")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "training_only": True,
            "complete": True,
            "fingerprint": dict(fingerprint),
            "content": dict(content),
        }
        payload["payload_sha256"] = payload_checksum(payload)
        with _exclusive_lock(lock_path):
            # 同一fingerprintの正常cacheはimmutableとして再利用する。
            if path.is_file():
                self.load(fingerprint)
                return path
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
            )
            os.close(file_descriptor)
            temporary = Path(temporary_name)
            try:
                with gzip.open(str(temporary), "wt", encoding="utf-8", compresslevel=6) as stream:
                    json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
                # rename前に一度読み戻し、partial writeを正規cacheへ昇格させない。
                with gzip.open(str(temporary), "rt", encoding="utf-8") as stream:
                    written = json.load(stream)
                if str(written.get("payload_sha256", "")) != payload_checksum(written):
                    raise RuntimeError("Exact Teacher Cache一時ファイルのchecksum不一致")
                os.replace(str(temporary), str(path))
            finally:
                if temporary.exists():
                    temporary.unlink()
        return path

