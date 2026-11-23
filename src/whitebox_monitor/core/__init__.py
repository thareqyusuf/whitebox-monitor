"""Framework-free core: types, prompt spans, positions, readouts, probe, injection."""

from .inject import inject
from .pipeline import compute_readout
from .positions import PositionError
from .probe import LinearProbe, load_probe, save_probe
from .prompt import build_prompt, fill_template
from .readouts import expected_score, parse_integer
from .types import (
    CaptureSpec,
    GenSpec,
    MonitorPrompt,
    PositionInfo,
    RawOutput,
    Readout,
)

__all__ = [
    "CaptureSpec", "GenSpec", "LinearProbe", "MonitorPrompt", "PositionError",
    "PositionInfo", "RawOutput", "Readout", "build_prompt", "compute_readout",
    "expected_score", "fill_template", "inject", "load_probe", "parse_integer",
    "save_probe",
]
