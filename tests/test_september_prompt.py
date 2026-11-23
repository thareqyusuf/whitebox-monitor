"""The vendored September prompt builder must not drift."""

import hashlib
import os
import sys
from pathlib import Path

import pytest
from september_fixture import GOLDEN, TRAJ

from whitebox_monitor.core import september as sep


def test_template_hash_is_septembers():
    text = sep.load_template()
    assert hashlib.sha256(text.encode()).hexdigest() == sep.TEMPLATE_SHA256


@pytest.mark.parametrize("idx", sorted(GOLDEN))
def test_golden_hashes_from_the_original_module(idx):
    prompt = sep.september_prompt(TRAJ, idx)
    assert sep.prompt_sha256(prompt) == GOLDEN[idx]


@pytest.mark.parametrize("idx", sorted(GOLDEN))
def test_spans(idx):
    prompt = sep.september_prompt(TRAJ, idx)
    assert prompt.messages == [{"role": "user", "content": prompt.content}]
    a0, a1 = prompt.spans["action"]
    h0, h1 = prompt.spans["history"]
    c = prompt.content
    assert c[:a0].endswith("<current_action>\n") and c[a1:].startswith("\n</current_action>")
    assert c[:h0].endswith("<history>\n") and c[h1:].startswith("\n</history>")
    assert prompt.action_text == sep.format_action(TRAJ["actions"][idx], hide_output=True)
    assert prompt.action_text.startswith("Tool call: ")


def test_missing_fields_raise():
    with pytest.raises(ValueError):
        sep.september_prompt({**TRAJ, "main_task_description": ""}, 0)
    with pytest.raises(ValueError):
        sep.september_prompt({k: v for k, v in TRAJ.items() if k != "side_task_goals"}, 0)
    with pytest.raises(IndexError):
        sep.september_prompt(TRAJ, 99)


def test_against_september_checkout_if_available():
    """Set WBM_SEPTEMBER_REPO to a checkout of linuxarena-monitor-readout to
    compare against the original module directly."""
    repo = os.environ.get("WBM_SEPTEMBER_REPO")
    if not repo:
        pytest.skip("WBM_SEPTEMBER_REPO not set")
    sys.path.insert(0, str(Path(repo) / "experiments" / "logit-curve-replication" / "gpu"))
    import format_monitor_prompt as fmp

    for idx in range(len(TRAJ["actions"])):
        assert sep.september_prompt(TRAJ, idx).content == fmp.build_prompt(TRAJ, idx)
