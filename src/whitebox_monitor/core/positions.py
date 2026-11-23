"""Map character spans of a :class:`MonitorPrompt` to token indices in the
fully rendered prompt (after the chat template).

Two rules, both recorded per position:

``<span>_last``  ("inside")      the last token that lies entirely inside the
                                 span. Text after the span cannot be part of
                                 it, so it is causally independent of anything
                                 that follows the span.
``<span>_end``   ("containing")  the token that contains the span's last
                                 character. This is September's rule
                                 (``monitor_prompt.tokenize_with_positions``).
                                 When the span's end is not a token boundary
                                 the token also covers text after the span: in
                                 all 2,012 September actions ``action_end`` is
                                 a token such as ``'}\\n`` whose newline
                                 belongs to the template.

The two agree exactly when the span ends on a token boundary
(``PositionInfo.clean``).

The tokenizer is duck-typed (Hugging Face fast-tokenizer interface); this
module imports no framework.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .types import MonitorPrompt, PositionInfo, Span

#: September position names -> span names.
SPAN_ALIASES = {"transcript": "history", "instruction": "message"}
LAST_PROMPT_TOKEN = ("prompt_end", "final")


class PositionError(ValueError):
    """A position could not be resolved. Never silently substituted."""


@dataclass(frozen=True)
class TokenizedPrompt:
    """A rendered prompt and its tokenization."""

    rendered: str
    input_ids: list[int]
    offsets: list[Span]
    content_start: int  # rendered char offset of the span message's content
    content_end: int


def last_token_inside(offsets: Sequence[Span], start: int, end: int) -> int:
    """Index of the last non-empty token with ``start <= tok_start`` and
    ``tok_end <= end``."""
    for i in range(len(offsets) - 1, -1, -1):
        tok_start, tok_end = offsets[i]
        if tok_end <= tok_start or tok_end > end:
            continue
        if tok_start < start:
            break
        return i
    raise PositionError(
        f"no token lies entirely inside the character span ({start}, {end})"
    )


def token_containing(offsets: Sequence[Span], char: int) -> int:
    """Index of the last non-empty token whose span contains ``char``."""
    for i in range(len(offsets) - 1, -1, -1):
        tok_start, tok_end = offsets[i]
        if tok_start <= char < tok_end:
            return i
    raise PositionError(f"no token contains character {char}")


def locate_content(tokenizer: Any, messages: Sequence[dict], index: int,
                   template_kwargs: Mapping[str, Any] | None = None) -> tuple[str, int]:
    """Render ``messages`` with the chat template and find where the content of
    ``messages[index]`` sits in the rendered string.

    The message boundary is found by rendering once more with a sentinel in
    place of the content, so text repeated elsewhere in the prompt cannot
    confuse it. Raises if the template alters the content (e.g. trims it).
    """
    kwargs = dict(template_kwargs or {})
    rendered = tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=True, **kwargs
    )
    content = messages[index]["content"]
    sentinel = f"⟪WBM-{secrets.token_hex(8)}⟫"
    probe = [dict(m) for m in messages]
    probe[index]["content"] = sentinel
    rendered_probe = tokenizer.apply_chat_template(
        probe, tokenize=False, add_generation_prompt=True, **kwargs
    )
    if rendered_probe.count(sentinel) != 1:
        raise PositionError(
            "the chat template does not render the message content exactly "
            f"once (found {rendered_probe.count(sentinel)} copies)"
        )
    start = rendered_probe.index(sentinel)
    expected = (
        rendered_probe[:start] + content + rendered_probe[start + len(sentinel):]
    )
    if rendered != expected:
        raise PositionError(
            "the chat template alters the message content (trimming or "
            "escaping); character spans cannot be mapped. Normalise the "
            "content before building the MonitorPrompt."
        )
    return rendered, start


def render_and_tokenize(tokenizer: Any, prompt: MonitorPrompt,
                        template_kwargs: Mapping[str, Any] | None = None) -> TokenizedPrompt:
    """Render with the chat template (untokenized), then tokenize the rendered
    string with offsets. The backend must feed exactly ``input_ids``."""
    rendered, start = locate_content(
        tokenizer, prompt.messages, prompt.span_message_index, template_kwargs
    )
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    if len(ids) != len(offsets) or not ids:
        raise PositionError("tokenizer returned no usable offset mapping")
    return TokenizedPrompt(
        rendered=rendered,
        input_ids=ids,
        offsets=offsets,
        content_start=start,
        content_end=start + len(prompt.content),
    )


def _span_for(name: str, prompt: MonitorPrompt) -> Span:
    name = SPAN_ALIASES.get(name, name)
    if name == "message":
        return (0, len(prompt.content))
    if name not in prompt.spans:
        raise PositionError(
            f"prompt has no span {name!r} (available: {sorted(prompt.spans)})"
        )
    return prompt.spans[name]


def resolve_position(name: str, tokenized: TokenizedPrompt, prompt: MonitorPrompt) -> PositionInfo:
    """Resolve one position name to a token index. Raises PositionError."""
    offsets = tokenized.offsets
    if name in LAST_PROMPT_TOKEN:
        i = len(offsets) - 1
        return PositionInfo(name, i, "last_prompt_token", offsets[i][0], offsets[i][1],
                            None, True)
    if name.endswith("_last"):
        rule, span_name = "inside", name[: -len("_last")]
    elif name.endswith("_end"):
        rule, span_name = "containing", name[: -len("_end")]
    else:
        raise PositionError(
            f"unknown position {name!r}: use '<span>_last', '<span>_end', "
            "'prompt_end' or 'final'"
        )
    start, end = _span_for(span_name, prompt)
    if start == end:
        raise PositionError(f"span for position {name!r} is empty")
    r_start = tokenized.content_start + start
    r_end = tokenized.content_start + end
    if rule == "inside":
        i = last_token_inside(offsets, r_start, r_end)
    else:
        i = token_containing(offsets, r_end - 1)
    return PositionInfo(name, i, rule, offsets[i][0], offsets[i][1], r_end,
                        offsets[i][1] == r_end)


def resolve_positions(names: Sequence[str], tokenized: TokenizedPrompt,
                      prompt: MonitorPrompt, tokenizer: Any = None) -> dict[str, PositionInfo]:
    """Resolve several positions; fill in token id and decoded token."""
    out: dict[str, PositionInfo] = {}
    for name in names:
        info = resolve_position(name, tokenized, prompt)
        token_id = tokenized.input_ids[info.token_index]
        decoded = tokenizer.decode([token_id]) if tokenizer is not None else None
        out[name] = PositionInfo(
            info.name, info.token_index, info.rule, info.char_start, info.char_end,
            info.span_char_end, info.clean, token_id, decoded,
        )
    return out


def shared_prefix_len(ids_a: Sequence[int], ids_b: Sequence[int]) -> int:
    """Number of leading token ids two tokenizations share.

    Use it to check that an injection after the action left the tokens up to
    the capture position untouched: ``shared_prefix_len(a, b) > index``.
    """
    n = 0
    for a, b in zip(ids_a, ids_b, strict=False):
        if a != b:
            break
        n += 1
    return n
