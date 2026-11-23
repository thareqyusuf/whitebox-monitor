"""Package versions recorded next to every number."""

from __future__ import annotations

from importlib import metadata

TRACKED = ("whitebox-monitor", "numpy", "safetensors", "torch", "transformers",
           "tokenizers", "control-arena", "inspect-ai", "vllm-lens", "vllm")


def package_versions() -> dict[str, str]:
    """Versions of the tracked packages that are installed."""
    out: dict[str, str] = {}
    for name in TRACKED:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out
