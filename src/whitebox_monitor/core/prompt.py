"""Build a :class:`MonitorPrompt` from a template, tracking character spans.

Spans come from offset arithmetic while the template is filled, never from
searching for the filled-in text afterwards: the same text can occur earlier
in the history.
"""

from __future__ import annotations

from collections.abc import Mapping
from string import Formatter

from .types import MonitorPrompt, Span


def fill_template(
    template: str,
    fields: Mapping[str, str],
    span_fields: Mapping[str, str],
) -> tuple[str, dict[str, Span]]:
    """``template.format(**fields)`` that also returns where fields landed.

    ``span_fields`` maps a template field name to the span name to record.
    Extra keys in ``fields`` are ignored, as ``str.format`` does. A span field
    must appear exactly once in the template.
    """
    out: list[str] = []
    pos = 0
    spans: dict[str, Span] = {}
    seen: dict[str, int] = {}
    for literal, name, spec, conversion in Formatter().parse(template):
        out.append(literal)
        pos += len(literal)
        if name is None:
            continue
        if spec or conversion:
            raise ValueError(
                f"field {{{name}}} uses a format spec or conversion, which "
                "span tracking does not support"
            )
        if name not in fields:
            raise KeyError(f"template field {{{name}}} has no value")
        value = fields[name]
        if not isinstance(value, str):
            raise TypeError(f"value for {{{name}}} must be str")
        seen[name] = seen.get(name, 0) + 1
        if name in span_fields:
            spans[span_fields[name]] = (pos, pos + len(value))
        out.append(value)
        pos += len(value)
    for name, span_name in span_fields.items():
        count = seen.get(name, 0)
        if count != 1:
            raise ValueError(
                f"span field {{{name}}} (span {span_name!r}) must appear "
                f"exactly once in the template, found {count}"
            )
    return "".join(out), spans


def build_prompt(
    template: str,
    fields: Mapping[str, str],
    *,
    action_field: str = "action",
    extra_span_fields: Mapping[str, str] | None = None,
    system: str | None = None,
) -> MonitorPrompt:
    """One user message from ``template``; ``action_field`` becomes the
    ``"action"`` span. Optionally prepend a system message."""
    span_fields = {action_field: "action", **(extra_span_fields or {})}
    text, spans = fill_template(template, fields, span_fields)
    messages: list[dict] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": text})
    return MonitorPrompt(
        messages=messages, spans=spans, span_message_index=len(messages) - 1
    )
