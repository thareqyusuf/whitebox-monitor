import pytest

from whitebox_monitor import MonitorPrompt, build_prompt
from whitebox_monitor.core.prompt import fill_template


def test_spans_come_from_offsets_not_search():
    action = "ls"
    text, spans = fill_template("{h} then {a} and {{literal}} {h2}",
                                {"h": "ls ls ls", "a": action, "h2": "ls"}, {"a": "action"})
    assert text == "ls ls ls then ls and {literal} ls"
    start, end = spans["action"]
    assert (start, end) == (14, 16) and text[start:end] == action
    assert text.find(action) != start  # a naive search would be wrong


def test_span_field_must_appear_once():
    with pytest.raises(ValueError, match="exactly once"):
        fill_template("{a} {a}", {"a": "x"}, {"a": "action"})
    with pytest.raises(ValueError, match="exactly once"):
        fill_template("nothing", {"a": "x"}, {"a": "action"})


def test_missing_field_and_format_spec_raise():
    with pytest.raises(KeyError):
        fill_template("{a} {b}", {"a": "x"}, {"a": "action"})
    with pytest.raises(ValueError, match="format spec"):
        fill_template("{a!r}", {"a": "x"}, {"a": "action"})


def test_build_prompt_with_system():
    p = build_prompt("do {action} now", {"action": "rm -rf /"}, system="be careful")
    assert [m["role"] for m in p.messages] == ["system", "user"]
    assert p.span_message_index == 1 and p.action_text == "rm -rf /"


def test_monitor_prompt_validation():
    with pytest.raises(ValueError, match="action"):
        MonitorPrompt([{"role": "user", "content": "x"}], {})
    with pytest.raises(ValueError, match="outside"):
        MonitorPrompt([{"role": "user", "content": "x"}], {"action": (0, 5)})
    with pytest.raises(ValueError, match="empty"):
        MonitorPrompt([{"role": "user", "content": "x"}], {"action": (1, 1)})
