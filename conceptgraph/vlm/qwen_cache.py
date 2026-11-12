from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CACHE_FORMAT_VERSION = "qwen-semantic-cache-v1"


@dataclass(frozen=True)
class CacheLookup:
    hit: bool
    key: str
    status: str
    payload: dict[str, Any] | None = None
    diagnostic: str | None = None


class QwenDiskCache:
    def __init__(self, root: str | Path, *, format_version: str = CACHE_FORMAT_VERSION):
        self.root = Path(root)
        self.format_version = str(format_version)
        self._write_lock = threading.Lock()

    def build_key(
        self,
        *,
        task: str,
        schema_version: str,
        model_id: str,
        adapter_id: str | None,
        normalized_input: Any,
        image_hashes: list[str] | None = None,
        object_version: str | None = None,
        candidates: Any = None,
        edge_version: str | None = None,
    ) -> str:
        material = {
            "adapter_id": adapter_id,
            "cache_format_version": self.format_version,
            "candidates": candidates,
            "edge_version": edge_version,
            "image_hashes": image_hashes or [],
            "model_id": model_id,
            "normalized_input": normalized_input,
            "object_version": object_version,
            "schema_version": schema_version,
            "task": task,
        }
        return sha256_bytes(canonical_json_bytes(material))

    def path_for_key(self, key: str) -> Path:
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("cache key must be a lowercase SHA-256 digest")
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> CacheLookup:
        path = self.path_for_key(key)
        if not path.exists():
            return CacheLookup(False, key, "miss")
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return CacheLookup(False, key, "corrupt", diagnostic=_safe_diagnostic(exc))
        if not isinstance(record, dict):
            return CacheLookup(False, key, "incomplete", diagnostic="cache record is not an object")
        if record.get("cache_format_version") != self.format_version:
            return CacheLookup(False, key, "version_mismatch", diagnostic="cache format version mismatch")
        required = {"cache_format_version", "key", "payload", "payload_sha256"}
        if not required.issubset(record):
            missing = sorted(required - set(record))
            return CacheLookup(False, key, "incomplete", diagnostic=f"missing fields: {missing}")
        if record["key"] != key or not isinstance(record["payload"], dict):
            return CacheLookup(False, key, "incomplete", diagnostic="cache key or payload is invalid")
        actual_digest = sha256_bytes(canonical_json_bytes(record["payload"]))
        if actual_digest != record["payload_sha256"]:
            return CacheLookup(False, key, "corrupt", diagnostic="payload checksum mismatch")
        return CacheLookup(True, key, "hit", payload=record["payload"])

    def put(self, key: str, payload: dict[str, Any]) -> Path:
        path = self.path_for_key(key)
        record = {
            "cache_format_version": self.format_version,
            "key": key,
            "payload": canonicalize(payload),
        }
        record["payload_sha256"] = sha256_bytes(canonical_json_bytes(record["payload"]))
        data = canonical_json_bytes(record) + b"\n"
        with self._write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary: Path | None = None
            try:
                descriptor, temp_name = tempfile.mkstemp(
                    prefix=f".{key}.", suffix=".tmp", dir=path.parent
                )
                temporary = Path(temp_name)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                temporary = None
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return path


def canonical_json_bytes(value: Any) -> bytes:
    normalized = canonicalize(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats cannot be canonicalized")
        return 0.0 if value == 0.0 else value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Path):
        return unicodedata.normalize("NFC", str(value))
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            string_key = unicodedata.normalize("NFC", str(key))
            if string_key in result:
                raise ValueError(f"duplicate canonical key: {string_key}")
            result[string_key] = canonicalize(value[key])
        return result
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, set):
        normalized = [canonicalize(item) for item in value]
        return sorted(normalized, key=lambda item: canonical_json_bytes(item))
    if hasattr(value, "tolist"):
        return canonicalize(value.tolist())
    raise TypeError(f"unsupported value for canonical JSON: {type(value).__name__}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_version(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _safe_diagnostic(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:240]}"
