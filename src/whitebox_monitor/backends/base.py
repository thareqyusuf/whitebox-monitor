"""The backend protocol. A backend turns one prompt into a RawOutput."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.types import CaptureSpec, GenSpec, MonitorPrompt, RawOutput


@runtime_checkable
class Backend(Protocol):
    """Runs the monitor on one prompt.

    One call returns the prompt-position residuals and the score-position
    distribution. A backend that needs more than one pass says so in its
    docstring and feeds byte-identical prompts to each pass.

    ``name`` goes into every ``Readout.meta`` as ``backend``.
    """

    name: str

    def run(self, prompt: MonitorPrompt, capture: CaptureSpec, gen: GenSpec) -> RawOutput:
        ...
