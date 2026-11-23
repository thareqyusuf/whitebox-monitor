"""Backend output -> the three readouts. No framework import."""

from __future__ import annotations

from typing import Any

from .probe import LinearProbe
from .readouts import expected_score, parse_integer
from .types import CaptureSpec, GenSpec, MonitorPrompt, RawOutput, Readout
from .versions import package_versions


def readout_from_raw(raw: RawOutput, gen: GenSpec, probe: LinearProbe | None = None) -> Readout:
    """Turn a :class:`RawOutput` into a :class:`Readout`.

    Nothing is guessed: an unparseable score gives ``integer=None``, missing
    probabilities give ``expected=None``, each with a reason in ``meta``.
    """
    lo, hi = min(gen.score_values), max(gen.score_values)
    integer, parse_info = parse_integer(raw.final_text, lo, hi)
    if raw.final_text is None and raw.meta.get("final_text_reason"):
        parse_info["reason"] = raw.meta["final_text_reason"]

    expected, probs, e_info = expected_score(raw.forced_score_probs)
    if raw.forced_score_probs is None and raw.meta.get("forced_reason"):
        e_info["reason"] = raw.meta["forced_reason"]
    expected_post, probs_post, post_info = expected_score(raw.post_score_probs)
    if raw.post_score_probs is None and raw.meta.get("post_reason"):
        post_info["reason"] = raw.meta["post_reason"]

    probe_score: float | None = None
    probe_meta: dict[str, Any] | None = None
    if probe is not None:
        probe_score = probe.apply(raw.residuals)
        probe_meta = probe.to_meta()

    meta: dict[str, Any] = dict(raw.meta)
    meta.update({
        "versions": package_versions(),
        "positions": {name: info.to_meta() for name, info in raw.positions.items()},
        "integer_parse": parse_info,
        "expected_info": e_info,
        "expected_post_info": post_info,
        "expected_source": "forced readout after " + repr(list(gen.forced_prefix)),
        "expected_post_source": "greedy step that emitted the parsed score",
        "score_values": list(gen.score_values),
        "probe": probe_meta,
        "residual_keys": [[layer, pos] for layer, pos in sorted(raw.residuals)],
    })
    for name, info in raw.positions.items():
        if name.startswith("action_"):
            meta.setdefault("action_token_index", info.token_index)
            meta.setdefault("action_token_position", name)
            break
    return Readout(
        integer=integer,
        expected=expected,
        score_token_probs=probs,
        residuals=raw.residuals,
        probe_score=probe_score,
        reasoning=raw.reasoning,
        completion=raw.completion,
        meta=meta,
        expected_post=expected_post,
        score_token_probs_post=probs_post,
    )


def compute_readout(backend: Any, prompt: MonitorPrompt, capture: CaptureSpec,
                    gen: GenSpec, probe: LinearProbe | None = None) -> Readout:
    """Run ``backend`` on ``prompt`` and compute the readouts.

    If a probe is given, its ``(layer, position)`` must be in ``capture``;
    this is checked before the model runs.
    """
    if probe is not None and (
        probe.layer not in capture.layers or probe.position not in capture.positions
    ):
        raise ValueError(
            f"probe needs layer {probe.layer} at {probe.position!r}, which the "
            f"CaptureSpec does not capture ({capture})"
        )
    raw = backend.run(prompt, capture, gen)
    return readout_from_raw(raw, gen, probe)
