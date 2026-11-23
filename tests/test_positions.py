"""Position index: the part most likely to be silently wrong."""

import pytest

from whitebox_monitor import MonitorPrompt, PositionError, build_prompt
from whitebox_monitor.core.positions import (
    last_token_inside,
    render_and_tokenize,
    resolve_position,
    resolve_positions,
    shared_prefix_len,
    token_containing,
)


# ── pure offset arithmetic ──────────────────────────────────────────────────
def test_action_ending_mid_token():
    # tokens: [0,4) [4,9) [9,14); the action is chars 4..11, ending mid-token
    offsets = [(0, 4), (4, 9), (9, 14)]
    assert last_token_inside(offsets, 4, 11) == 1      # last token fully inside
    assert token_containing(offsets, 10) == 2          # September: straddles


def test_trailing_whitespace_merged_into_next_token():
    # "...}" + "\n</": the tokenizer merged "}\n" into one token [8,10)
    offsets = [(0, 4), (4, 8), (8, 10), (10, 12)]
    action_end = 9  # the action ends after "}", before "\n"
    assert last_token_inside(offsets, 0, action_end) == 1
    assert token_containing(offsets, action_end - 1) == 2


def test_clean_boundary_rules_agree():
    offsets = [(0, 4), (4, 8), (8, 10)]
    assert last_token_inside(offsets, 0, 8) == token_containing(offsets, 7) == 1


def test_zero_length_special_tokens_are_skipped():
    offsets = [(0, 0), (0, 4), (4, 8), (0, 0)]
    assert last_token_inside(offsets, 0, 8) == 2
    assert token_containing(offsets, 7) == 2


def test_no_token_inside_raises():
    with pytest.raises(PositionError):
        last_token_inside([(0, 10)], 2, 5)
    with pytest.raises(PositionError):
        token_containing([(0, 3)], 7)


# ── with a real tokenizer and chat template ─────────────────────────────────
def _check(tok, prompt, name="action_last"):
    tp = render_and_tokenize(tok, prompt)
    info = resolve_positions([name], tp, prompt, tok)[name]
    a_start = tp.content_start + prompt.spans["action"][0]
    a_end = tp.content_start + prompt.spans["action"][1]
    assert tp.rendered[a_start:a_end] == prompt.action_text
    return tp, info, a_start, a_end


def test_action_last_is_last_token_of_action(tokenizer, make_prompt):
    for action in ["Tool call: bash: {'cmd': 'cat /etc/passwd'}",
                   "Tool call: bash: {'cmd': 'ls'} ",          # trailing space
                   "Tool call: submit: {}",
                   "plain words with no brace"]:
        prompt = make_prompt(action=action)
        tp, info, a_start, a_end = _check(tokenizer, prompt)
        # the token lies inside the action and reaches as far as a whole token can
        assert a_start <= info.char_start and info.char_end <= a_end
        assert prompt.action_text.endswith(tp.rendered[info.char_start:info.char_end]) or not info.clean
        # the next token is not inside the action
        nxt = tp.offsets[info.token_index + 1]
        assert nxt[1] > a_end
        assert info.decoded == tokenizer.decode([tp.input_ids[info.token_index]])


def test_merged_newline_token_straddles_like_september(tokenizer, make_prompt):
    """`}` + template newline is one token: `action_end` (September) picks it,
    `action_last` picks the token before, and neither is silently swapped."""
    prompt = make_prompt(action="Tool call: bash: {'cmd': 'cat /etc/passwd'}")
    tp, last, a_start, a_end = _check(tokenizer, prompt, "action_last")
    _, end, _, _ = _check(tokenizer, prompt, "action_end")
    assert end.decoded == "}\n" and not end.clean and end.char_end == a_end + 1
    assert end.rule == "containing" and last.rule == "inside"
    assert last.token_index == end.token_index - 1
    assert not last.clean and last.char_end == a_end - 1


def test_clean_boundary_same_token(tokenizer, make_prompt):
    prompt = make_prompt(action="plain words with no brace")
    _, last, _, _ = _check(tokenizer, prompt, "action_last")
    _, end, _, _ = _check(tokenizer, prompt, "action_end")
    assert last.token_index == end.token_index and last.clean and end.clean


def test_repeated_action_text_resolves_to_the_action_block(tokenizer, make_prompt):
    action = "Tool call: bash: {'cmd': 'pwd'}"
    history = f"Action 0:\n{action}\nResult: /app\n\nAction 1:\n{action}\nResult: /app"
    prompt = make_prompt(action=action, history=history, system=action)
    tp, info, a_start, a_end = _check(tokenizer, prompt)
    assert tp.rendered.count(action) == 4
    assert tp.rendered.rfind(action) == a_start         # the last copy is the action
    assert tp.rendered.find(action) < a_start
    assert a_start <= info.char_start < a_end
    hist = resolve_position("history_last", tp, prompt)
    assert hist.token_index < info.token_index


def test_september_position_names(tokenizer, make_prompt):
    prompt = make_prompt()
    tp = render_and_tokenize(tokenizer, prompt)
    pos = resolve_positions(["transcript_end", "action_end", "instruction_end", "prompt_end", "final"],
                            tp, prompt, tokenizer)
    assert pos["prompt_end"].token_index == pos["final"].token_index == len(tp.input_ids) - 1
    assert pos["transcript_end"].token_index < pos["action_end"].token_index \
        < pos["instruction_end"].token_index < pos["prompt_end"].token_index
    assert pos["instruction_end"].char_end == tp.content_end


def test_unknown_position_and_span_raise(tokenizer, make_prompt):
    prompt = make_prompt()
    tp = render_and_tokenize(tokenizer, prompt)
    with pytest.raises(PositionError, match="unknown position"):
        resolve_position("action", tp, prompt)
    with pytest.raises(PositionError, match="no span"):
        resolve_position("nonexistent_last", tp, prompt)


def test_template_that_trims_content_raises(trimming_tokenizer):
    prompt = build_prompt("  padded {action}\n", {"action": "ls"})
    with pytest.raises(PositionError, match="alters the message content"):
        render_and_tokenize(trimming_tokenizer, prompt)


def test_span_in_second_message(tokenizer):
    prompt = MonitorPrompt(
        messages=[{"role": "system", "content": "run ls"}, {"role": "user", "content": "run ls"}],
        spans={"action": (4, 6)}, span_message_index=1)
    tp = render_and_tokenize(tokenizer, prompt)
    info = resolve_position("action_last", tp, prompt)
    assert info.char_end == tp.rendered.rfind("ls") + 2


def test_shared_prefix_len():
    assert shared_prefix_len([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert shared_prefix_len([1, 2], [1, 2, 3]) == 2
