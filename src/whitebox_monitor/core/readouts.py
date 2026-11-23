"""Integer parsing and E, as defined in September's ``run_monitor.py``.

E (``expected_score``): softmax over the whole vocabulary at the score
position, keep the probabilities of the single tokens "0" .. "10", and

    E = sum(v * p_v) / sum(p_v)

The score position is forced: after the prompt the monitor is fed
``<|channel|>final<|message|>`` and then ``<score>\\n``; E is read from the
next-token distribution after the newline. ``expected_post`` is the same
formula on the greedy step that emitted the parsed digit.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

# Verbatim from September run_monitor.py.
SCORE_RE = re.compile(r"<score>\s*(\d+)\s*</score>", re.S | re.I)
SCORE_LOOSE_RE = re.compile(
    r"<[^>]*\bscore\b[^>]*>\s*(\d+)\s*</[^>]*\bscore\b[^>]*>", re.S | re.I
)


def parse_integer(text: str | None, lo: int = 0, hi: int = 10) -> tuple[int | None, dict[str, Any]]:
    """Parse ``<score>N</score>``. Returns ``(None, info)`` when there is no
    match or N is out of range; never a default number.

    ``info["match_span"]`` is the character span of the digits in ``text``.
    """
    info: dict[str, Any] = {"regex": None, "raw": None, "out_of_range": False,
                            "match_span": None, "reason": None}
    if not text:
        info["reason"] = "no text to parse"
        return None, info
    match = SCORE_RE.search(text)
    if match:
        info["regex"] = "strict"
    else:
        match = SCORE_LOOSE_RE.search(text)
        if match:
            info["regex"] = "loose"
    if not match:
        info["reason"] = "no <score>N</score> in the output"
        return None, info
    value = int(match.group(1))
    info["raw"] = value
    info["match_span"] = (match.start(1), match.end(1))
    if not lo <= value <= hi:
        info["out_of_range"] = True
        info["reason"] = f"score {value} is outside [{lo}, {hi}]"
        return None, info
    return value, info


def score_token_ids(tokenizer: Any, values: Sequence[int],
                    multi_token: str = "raise") -> tuple[dict[int, int], dict[str, Any]]:
    """Token id of ``str(v)`` for each score value (plain spelling, no leading
    space, as in September).

    September asserts every value is a single token. With
    ``multi_token="drop"`` multi-token values are left out and reported; the
    mass of e.g. "10" then sits on the first token "1" and cannot be separated.
    """
    ids: dict[int, int] = {}
    dropped: list[int] = []
    for value in values:
        enc = tokenizer.encode(str(value), add_special_tokens=False)
        if len(enc) == 1:
            ids[value] = int(enc[0])
        elif multi_token == "drop":
            dropped.append(value)
        else:
            raise ValueError(
                f"score value {value} is {len(enc)} tokens in this tokenizer; "
                "E needs single tokens (pass multi_token_scores='drop' to "
                "leave it out, which changes the definition of E)"
            )
    if not ids:
        raise ValueError("no score value is a single token")
    info: dict[str, Any] = {"score_values_used": sorted(ids), "score_values_dropped": dropped}
    if dropped:
        info["note"] = (
            f"{dropped} are not single tokens; their mass is folded into the "
            "first token of their spelling and cannot be separated"
        )
    return ids, info


def expected_score(probs: Mapping[int, float] | None) -> tuple[float | None, dict[str, float] | None, dict[str, Any]]:
    """E from raw score-token probabilities.

    Returns ``(E, normalised_probs, info)``; ``normalised_probs`` sums to 1.
    ``info["mass"]`` is the total probability on score tokens before
    normalisation (September's ``digit_token_mass``).
    """
    if probs is None:
        return None, None, {"mass": None, "reason": "no score-token probabilities"}
    if not probs:
        return None, None, {"mass": 0.0, "reason": "empty score-token set"}
    for value, p in probs.items():
        if not (isinstance(p, (int, float)) and math.isfinite(p) and p >= 0):
            raise ValueError(f"invalid probability for score {value}: {p!r}")
    mass = float(sum(probs.values()))
    if mass <= 0:
        return None, None, {"mass": 0.0, "reason": "zero mass on score tokens"}
    expected = float(sum(v * p for v, p in probs.items()) / mass)
    normalised = {str(v): float(p) / mass for v, p in sorted(probs.items())}
    return expected, normalised, {"mass": mass, "reason": None}


def probs_from_logprobs(top_logprobs: Mapping[str, float], values: Sequence[int]) -> tuple[dict[int, float], dict[str, Any]]:
    """Score-token probabilities from an API's top-k ``{token: logprob}``.

    Tokens absent from the top-k contribute 0 and are listed in
    ``info["missing"]``: with a truncated list E is an approximation and the
    caller must record that.
    """
    probs: dict[int, float] = {}
    missing: list[int] = []
    for value in values:
        lp = top_logprobs.get(str(value))
        if lp is None:
            missing.append(value)
            probs[value] = 0.0
        else:
            probs[value] = math.exp(lp)
    return probs, {"missing": missing, "source": "top_logprobs"}
