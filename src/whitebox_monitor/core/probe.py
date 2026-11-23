"""Load and apply a frozen linear probe. Training lives in ``steered-verdict``.

``.npz`` layout:

    weights            float [d]
    bias               float scalar
    layer              int scalar      (0 = embedding output, as CaptureSpec)
    position           str             (position name, e.g. "action_last")
    manifest_sha256    str             (hash of the training manifest)
    scaler_mean        float [d]       optional
    scaler_scale       float [d]       optional (both or neither)

``score = weights . ((x - scaler_mean) / scaler_scale) + bias`` in float64.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED = ("weights", "bias", "layer", "position", "manifest_sha256")


@dataclass(frozen=True)
class LinearProbe:
    weights: np.ndarray
    bias: float
    layer: int
    position: str
    manifest_sha256: str
    scaler_mean: np.ndarray | None = None
    scaler_scale: np.ndarray | None = None
    file_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.weights.ndim != 1:
            raise ValueError("probe weights must be one-dimensional")
        if (self.scaler_mean is None) != (self.scaler_scale is None):
            raise ValueError("scaler_mean and scaler_scale must come together")
        if self.scaler_mean is not None and self.scaler_scale is not None:
            if self.scaler_mean.shape != self.weights.shape or \
                    self.scaler_scale.shape != self.weights.shape:
                raise ValueError("scaler shapes do not match the weights")
            if np.any(self.scaler_scale == 0):
                raise ValueError("scaler_scale contains zeros")

    @property
    def key(self) -> tuple[int, str]:
        return (self.layer, self.position)

    def apply_vector(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != self.weights.shape:
            raise ValueError(
                f"residual has shape {x.shape}, probe expects {self.weights.shape}"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("residual contains non-finite values")
        if self.scaler_mean is not None and self.scaler_scale is not None:
            x = (x - self.scaler_mean) / self.scaler_scale
        return float(x @ self.weights.astype(np.float64) + self.bias)

    def apply(self, residuals: Mapping[tuple[int, str], np.ndarray]) -> float:
        """Apply to a ``{(layer, position): vector}`` mapping. Raises KeyError
        if the probe's layer/position was not captured."""
        if self.key not in residuals:
            raise KeyError(
                f"probe needs residual {self.key}, captured: {sorted(residuals)}"
            )
        return self.apply_vector(residuals[self.key])

    def to_meta(self) -> dict[str, Any]:
        return {"layer": self.layer, "position": self.position,
                "manifest_sha256": self.manifest_sha256,
                "file_sha256": self.file_sha256,
                "has_scaler": self.scaler_mean is not None,
                "dim": int(self.weights.shape[0])}


def save_probe(path: str | Path, probe: LinearProbe) -> None:
    arrays: dict[str, Any] = {
        "weights": probe.weights, "bias": np.float64(probe.bias),
        "layer": np.int64(probe.layer), "position": np.str_(probe.position),
        "manifest_sha256": np.str_(probe.manifest_sha256),
    }
    if probe.scaler_mean is not None and probe.scaler_scale is not None:
        arrays["scaler_mean"] = probe.scaler_mean
        arrays["scaler_scale"] = probe.scaler_scale
    with open(path, "wb") as f:
        np.savez(f, **arrays)


def load_probe(path: str | Path) -> LinearProbe:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"probe file not found: {path}")
    data = path.read_bytes()
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in REQUIRED if k not in z.files]
        if missing:
            raise ValueError(f"probe {path} is missing fields: {missing}")
        return LinearProbe(
            weights=np.asarray(z["weights"]),
            bias=float(z["bias"]),
            layer=int(z["layer"]),
            position=str(z["position"]),
            manifest_sha256=str(z["manifest_sha256"]),
            scaler_mean=np.asarray(z["scaler_mean"]) if "scaler_mean" in z.files else None,
            scaler_scale=np.asarray(z["scaler_scale"]) if "scaler_scale" in z.files else None,
            file_sha256=hashlib.sha256(data).hexdigest(),
        )
