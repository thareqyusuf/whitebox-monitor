"""Framework-free data types shared by every layer.

Nothing in ``core`` imports Inspect, ControlArena, vLLM or torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

Span = tuple[int, int]

#: Score values of the September monitor prompt ("a number from 0 to 10").
DEFAULT_SCORE_VALUES: tuple[int, ...] = tuple(range(11))


@dataclass(frozen=True)
class MonitorPrompt:
    """Chat messages sent to the monitor, plus named character spans.

    ``spans`` are ``(start, end)`` character offsets into the *content* of
    ``messages[span_message_index]`` (end exclusive). ``"action"`` is required:
    it is the text of the action under review.
    """

    messages: list[dict]
    spans: dict[str, Span]
    span_message_index: int = 0

    def __post_init__(self) -> None:
        if "action" not in self.spans:
            raise ValueError('MonitorPrompt.spans must include "action"')
        if not 0 <= self.span_message_index < len(self.messages):
            raise ValueError("span_message_index is out of range")
        content = self.messages[self.span_message_index].get("content")
        if not isinstance(content, str):
            raise ValueError("the span-holding message must have string content")
        for name, (start, end) in self.spans.items():
            if not 0 <= start <= end <= len(content):
                raise ValueError(
                    f"span {name!r}=({start}, {end}) is outside the message "
                    f"content (length {len(content)})"
                )
        start, end = self.spans["action"]
        if start == end:
            raise ValueError('the "action" span is empty')

    @property
    def content(self) -> str:
        """Content of the message that holds the spans."""
        return self.messages[self.span_message_index]["content"]

    def span_text(self, name: str) -> str:
        start, end = self.spans[name]
        return self.content[start:end]

    @property
    def action_text(self) -> str:
        return self.span_text("action")


@dataclass(frozen=True)
class CaptureSpec:
    """Which residuals to capture.

    ``layers`` index the residual stream as September did: ``0`` is the
    embedding output and ``i`` is the output of decoder block ``i - 1``.
    ``positions`` are position names resolved by ``core.positions``
    (``"action_last"``, ``"action_end"``, ``"prompt_end"``, ...).
    """

    layers: tuple[int, ...] = ()
    positions: tuple[str, ...] = ("action_last",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "positions", tuple(self.positions))
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("duplicate layers in CaptureSpec")
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("duplicate positions in CaptureSpec")
        if any(layer < 0 for layer in self.layers):
            raise ValueError("layers must be non-negative")


@dataclass(frozen=True)
class GenSpec:
    """How the monitor is read out.

    ``forced_prefix`` is the text force-fed after the prompt to reach the
    score position for E. Each segment is tokenized and fed on its own, as
    September did (``<|channel|>final<|message|>`` then ``<score>\\n``).
    ``generate=False`` skips the unforced generation, so ``integer`` and
    ``expected_post`` are ``None``.
    """

    max_new_tokens: int = 1024
    forced_prefix: tuple[str, ...] = ("<score>\n",)
    score_values: tuple[int, ...] = DEFAULT_SCORE_VALUES
    generate: bool = True
    #: "raise": every score value must be a single token (September).
    #: "drop": drop multi-token values and say so in ``meta``.
    multi_token_scores: str = "raise"

    def __post_init__(self) -> None:
        object.__setattr__(self, "forced_prefix", tuple(self.forced_prefix))
        object.__setattr__(self, "score_values", tuple(self.score_values))
        if self.multi_token_scores not in ("raise", "drop"):
            raise ValueError('multi_token_scores must be "raise" or "drop"')
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")


@dataclass(frozen=True)
class PositionInfo:
    """A resolved capture position in the fully rendered, tokenized prompt."""

    name: str
    token_index: int
    rule: str  # "inside" | "containing" | "last_prompt_token"
    char_start: int  # token's character span in the rendered prompt
    char_end: int
    span_char_end: int | None  # rendered char offset where the span ends
    #: True when the token ends exactly where the span ends.
    clean: bool
    token_id: int | None = None
    decoded: str | None = None

    def to_meta(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "token_index": self.token_index,
            "rule": self.rule,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "span_char_end": self.span_char_end,
            "clean": self.clean,
            "token_id": self.token_id,
            "decoded": self.decoded,
        }


@dataclass
class RawOutput:
    """What a backend returns for one prompt.

    Probabilities are raw next-token probabilities (softmax over the whole
    vocabulary) of the score tokens, keyed by score value; they do not sum to
    one. ``None`` always comes with a reason in ``meta``.
    """

    completion: str
    final_text: str | None  # the part of the completion the score is parsed from
    reasoning: str | None
    forced_score_probs: dict[int, float] | None
    post_score_probs: dict[int, float] | None
    residuals: dict[tuple[int, str], np.ndarray]
    positions: dict[str, PositionInfo]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Readout:
    """The three readouts for one action.

    ``expected`` is September's E: the forced readout. ``expected_post`` is
    September's ``expected_score_post``: the distribution of the greedy step
    that emitted the parsed digit.
    """

    integer: int | None
    expected: float | None
    score_token_probs: dict[str, float] | None  # normalised, sums to 1
    residuals: dict[tuple[int, str], np.ndarray]
    probe_score: float | None
    reasoning: str | None
    completion: str
    meta: dict[str, Any]
    expected_post: float | None = None
    score_token_probs_post: dict[str, float] | None = None
