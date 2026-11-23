"""Residuals on disk, outside the Inspect log.

Layout of a store directory::

    index.jsonl               one row per residual vector
    shard-<writer>-00000.safetensors
    ...

Each row is keyed by ``(run_id, sample_id, message_id, layer, position)`` and
records the shard, the tensor name inside it, the stored dtype and the shape.
Vectors are stored as float16 unless told otherwise. A shard is written
before its index rows, so the index never points at a missing tensor.

Several writers may share a directory: shard names carry a per-writer id and
index rows are appended under a lock in single ``write`` calls.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

INDEX_NAME = "index.jsonl"
KEY_FIELDS = ("run_id", "sample_id", "message_id", "layer", "position")
Key = tuple[str, str, str, int, str]


def make_key(run_id: str, sample_id: str | int, message_id: str, layer: int, position: str) -> Key:
    return (str(run_id), str(sample_id), str(message_id), int(layer), str(position))


class SidecarWriter:
    """Buffer vectors and flush them to safetensors shards."""

    def __init__(self, root: str | Path, dtype: str = "float16", shard_size: int = 256) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dtype = np.dtype(dtype)
        if self.dtype.kind != "f":
            raise ValueError("sidecar dtype must be a float type")
        self.shard_size = shard_size
        self.writer_id = uuid.uuid4().hex[:8]
        self._lock = threading.Lock()
        self._buffer: list[tuple[Key, np.ndarray, dict[str, Any]]] = []
        self._buffered_keys: set[Key] = set()
        self._shard_no = 0
        self._written: set[Key] = {row_key(r) for r in iter_index(self.root)}

    def add(self, run_id: str, sample_id: str | int, message_id: str,
            residuals: Mapping[tuple[int, str], np.ndarray],
            extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Queue every ``(layer, position)`` vector of one action. Returns a
        JSON-able pointer for the log. Duplicate keys are an error."""
        keys: list[Key] = []
        with self._lock:
            for (layer, position), vec in sorted(residuals.items()):
                key = make_key(run_id, sample_id, message_id, layer, position)
                if key in self._written or key in self._buffered_keys:
                    raise KeyError(f"sidecar key already written: {key}")
                arr = np.asarray(vec)
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"non-finite residual for {key}")
                with np.errstate(over="ignore"):
                    stored = arr.astype(self.dtype)
                if not np.all(np.isfinite(stored)):
                    raise ValueError(
                        f"residual for {key} overflows {self.dtype}; use a wider dtype"
                    )
                self._buffer.append((key, stored, {"source_dtype": str(arr.dtype),
                                                   **dict(extra or {})}))
                self._buffered_keys.add(key)
                keys.append(key)
            if len(self._buffer) >= self.shard_size:
                self._flush_locked()
        return {
            "store": str(self.root),
            "run_id": str(run_id), "sample_id": str(sample_id),
            "message_id": str(message_id),
            "keys": [[k[3], k[4]] for k in keys],
            "dtype": str(self.dtype),
        }

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        name = f"shard-{self.writer_id}-{self._shard_no:05d}.safetensors"
        path = self.root / name
        tensors = {f"t{i:06d}": arr for i, (_k, arr, _e) in enumerate(self._buffer)}
        tmp = path.with_suffix(".safetensors.tmp")
        save_file(tensors, str(tmp))
        os.replace(tmp, path)
        lines = []
        for i, (key, arr, extra) in enumerate(self._buffer):
            row = dict(zip(KEY_FIELDS, key, strict=True))
            row.update({"shard": name, "tensor": f"t{i:06d}", "dtype": str(arr.dtype),
                        "shape": list(arr.shape), **extra})
            lines.append(json.dumps(row, ensure_ascii=False))
        with open(self.root / INDEX_NAME, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._written.update(k for k, _a, _e in self._buffer)
        self._buffer.clear()
        self._buffered_keys.clear()
        self._shard_no += 1

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> SidecarWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def iter_index(root: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(root) / INDEX_NAME
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def row_key(row: Mapping[str, Any]) -> Key:
    return make_key(*(row[f] for f in KEY_FIELDS))


class SidecarReader:
    """Read vectors back by key. Arrays come back in their stored dtype."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not (self.root / INDEX_NAME).is_file():
            raise FileNotFoundError(f"no sidecar index at {self.root / INDEX_NAME}")
        self.rows: dict[Key, dict[str, Any]] = {}
        for row in iter_index(self.root):
            key = row_key(row)
            if key in self.rows:
                raise ValueError(f"duplicate key in sidecar index: {key}")
            self.rows[key] = row
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def keys(self) -> list[Key]:
        return sorted(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __contains__(self, key: object) -> bool:
        return key in self.rows

    def get(self, run_id: str, sample_id: str | int, message_id: str, layer: int,
            position: str) -> np.ndarray:
        key = make_key(run_id, sample_id, message_id, layer, position)
        if key not in self.rows:
            raise KeyError(f"not in sidecar: {key}")
        row = self.rows[key]
        shard = row["shard"]
        if shard not in self._cache:
            self._cache = {shard: load_file(str(self.root / shard))}
        arr = self._cache[shard][row["tensor"]]
        if str(arr.dtype) != row["dtype"] or list(arr.shape) != row["shape"]:
            raise ValueError(f"sidecar entry {key} does not match its index row")
        return arr

    def get_action(self, run_id: str, sample_id: str | int, message_id: str) -> dict[tuple[int, str], np.ndarray]:
        """All ``(layer, position)`` vectors of one action."""
        prefix = (str(run_id), str(sample_id), str(message_id))
        out = {(k[3], k[4]): self.get(*k) for k in self.rows if k[:3] == prefix}
        if not out:
            raise KeyError(f"no residuals for {prefix}")
        return out
