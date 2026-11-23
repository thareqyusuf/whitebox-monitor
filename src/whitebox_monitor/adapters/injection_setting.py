"""Rollout-level injection setting for ControlArena. STUB: not built.

Static injection into a finished monitor prompt is ``core.inject``. Planting
text inside the sandbox during the rollout (README, config, tool output) is a
later phase; the design is in ``docs/injection-setting.md``.
"""

from __future__ import annotations


def injection_setting_builder(*args: object, **kwargs: object) -> None:
    raise NotImplementedError(
        "the rollout-level injection setting is not built yet; see "
        "docs/injection-setting.md. For static injection use whitebox_monitor.inject."
    )
