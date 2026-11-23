"""Static text injection into a monitor prompt. Pure function.

    before_action   <text><sep><action>      action text unchanged
    inside_action   <action[:k]><sep><text><action[k:]>, k = len(action) by
                    default; the action span grows to cover the injection
    after_action    <action><sep><text>      action text and span unchanged

Every span is shifted or extended so position names still resolve, and the
injected text is recorded as the ``"injection"`` span.

Tokenization caveat: ``after_action`` leaves the action's characters alone,
but a tokenizer may merge the action's last characters with what follows, so
check ``positions.shared_prefix_len`` (or ``PositionInfo.token_id``) when
comparing residuals across injected and clean prompts.
"""

from __future__ import annotations

from .types import MonitorPrompt, Span

PLACEMENTS = ("before_action", "inside_action", "after_action")


def _shift(span: Span, point: int, length: int) -> Span:
    """Span after inserting ``length`` characters at ``point``."""
    start, end = span
    if point <= start:
        return (start + length, end + length)
    if point < end:
        return (start, end + length)
    return span


def inject(prompt: MonitorPrompt, text: str, placement: str, *, sep: str = "\n",
           offset: int | None = None) -> MonitorPrompt:
    """Return a new prompt with ``text`` injected; ``prompt`` is not modified.

    ``offset`` (``inside_action`` only) is the character offset inside the
    action where the injection goes; default is the end of the action.
    """
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
    if not text:
        raise ValueError("injection text is empty")
    if "injection" in prompt.spans:
        raise ValueError("prompt already has an 'injection' span")
    if offset is not None and placement != "inside_action":
        raise ValueError("offset only applies to inside_action")

    content = prompt.content
    a_start, a_end = prompt.spans["action"]
    if placement == "before_action":
        point, inserted, inj_start = a_start, text + sep, a_start
    elif placement == "after_action":
        point, inserted, inj_start = a_end, sep + text, a_end + len(sep)
    else:
        k = a_end - a_start if offset is None else offset
        if not 0 <= k <= a_end - a_start:
            raise ValueError("offset is outside the action")
        point, inserted, inj_start = a_start + k, sep + text, a_start + k + len(sep)

    new_content = content[:point] + inserted + content[point:]
    length = len(inserted)
    spans: dict[str, Span] = {}
    for name, span in prompt.spans.items():
        if name == "action":
            continue
        spans[name] = _shift(span, point, length)
    if placement == "before_action":
        spans["action"] = (a_start + length, a_end + length)
    elif placement == "after_action":
        spans["action"] = (a_start, a_end)
    else:
        spans["action"] = (a_start, a_end + length)
    spans["injection"] = (inj_start, inj_start + len(text))

    messages = [dict(m) for m in prompt.messages]
    messages[prompt.span_message_index]["content"] = new_content
    return MonitorPrompt(messages=messages, spans=spans,
                         span_message_index=prompt.span_message_index)
