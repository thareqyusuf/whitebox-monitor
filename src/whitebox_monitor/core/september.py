"""September's monitor prompt, with character spans.

The formatting functions between the two "vendored" markers are copied
unchanged from the September experiment
(``thareqyusuf/linuxarena-monitor-readout`` at ``788c25f``,
``experiments/logit-curve-replication/gpu/format_monitor_prompt.py``), which
ports control-tower 1.0.1's live action-monitor prompt and was verified there
byte-identical on 17,234 actions. ``data/old_action.md`` is a byte copy of
the prompt template (sha256 ``a4a48d03...2058``, checked on load).

Do not edit the vendored block or the template: E and the prompt must match
September. ``tests/test_september_prompt.py`` pins golden hashes.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from importlib import resources
from pathlib import Path
from typing import Any

from .prompt import fill_template
from .types import MonitorPrompt

TEMPLATE_SHA256 = "a4a48d036bca6728925ad60061e214177e16fc063075ba084726ee52529d2058"
SEPTEMBER_COMMIT = "788c25f3ab8ab6b00b272367d7f61796d26e82ce"

# ── vendored from September: begin ──────────────────────────────────────────
# fmt: off
# ruff: noqa
DEFAULT_OUTPUT_LIMIT = 400000
HIDDEN_TOOLS = {"run_side_task_test", "submit", "side_task_scorer"}

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[ ]?(.*)")


# ── format_history.py ────────────────────────────────────────────────────────

def is_action_malformed(action: dict) -> bool:
    return bool((action.get("output_error") or "").startswith("parsing:"))


def is_action_hidden(action: dict) -> bool:
    return action.get("function") in HIDDEN_TOOLS


def should_skip_action(action: dict) -> str | None:
    if is_action_hidden(action):
        return f'"{action.get("function")}" tool calls are ignored'
    if is_action_malformed(action):
        return "malformed action (parsing error)"
    return None


def _format_action_call(action: dict, *, redact_str_replace_args: bool = False) -> str:
    function = action.get("function") or ""
    arguments = action.get("arguments")
    if arguments is None:
        arguments = {}
    if (
        redact_str_replace_args
        and function == "text_editor"
        and arguments.get("command") == "str_replace"
    ):
        redacted = {k: "[...]" if k in ("old_str", "new_str") else v for k, v in arguments.items()}
        return f"{function}: {redacted}"
    return f"{function}: {arguments}"


def _format_action_output(action: dict) -> str | None:
    output = action.get("output")
    if action.get("function") == "text_editor":
        # Reconstruct a diff from str_replace arguments when the output is empty.
        return output or format_str_replace_action_output(action.get("arguments") or {})
    if output is None:
        return None
    if len(output) > DEFAULT_OUTPUT_LIMIT:
        output = output[:DEFAULT_OUTPUT_LIMIT] + "... [truncated]"
    return output


def format_action(action: dict, hide_output: bool = False) -> str:
    action_output: str | None = None
    if not hide_output:
        action_output = _format_action_output(action)
    action_str = f"Tool call: {_format_action_call(action, redact_str_replace_args=bool(action_output))}"
    if hide_output:
        return action_str
    if action_output:
        action_str += f"\nResult: {action_output}"
    if action.get("output_error"):
        action_str += f"\nError: {action.get('output_error')}"
    return action_str


def format_history(actions: list[dict]) -> str:
    actions = [a for a in actions if not should_skip_action(a)]
    parts = [f"Action {i}:\n{format_action(a)}" for i, a in enumerate(actions)]
    return "\n\n".join(parts) if parts else "No previous actions."


# ── text_editor_output.py (unidiff re-implemented) ───────────────────────────

def _compute_diff_unidiff(old_str: str, new_str: str) -> str | None:
    diff_text = "\n".join(
        difflib.unified_diff(
            old_str.splitlines(), new_str.splitlines(),
            fromfile="old_version", tofile="edited_version", lineterm="",
        )
    )
    if not diff_text.strip():
        return None
    try:
        result: list[str] = []
        lines = diff_text.split("\n")
        i = 0
        # file header
        source_file = target_file = None
        while i < len(lines):
            ln = lines[i]
            if ln.startswith("--- "):
                source_file = ln[4:]
            elif ln.startswith("+++ "):
                target_file = ln[4:]
                i += 1
                break
            i += 1
        if source_file is None or target_file is None:
            raise ValueError("no file header")
        result.append(f"--- {source_file}")
        result.append(f"+++ {target_file}")
        while i < len(lines):
            m = _HUNK_RE.match(lines[i])
            if not m:
                raise ValueError(f"hunk header expected: {lines[i]!r}")
            s_start = int(m.group(1)); s_len = int(m.group(2)) if m.group(2) is not None else 1
            t_start = int(m.group(3)); t_len = int(m.group(4)) if m.group(4) is not None else 1
            result.append(f"@@ -{s_start},{s_len} +{t_start},{t_len} @@")
            i += 1
            src, tgt = s_start, t_start
            n_src = n_tgt = 0
            while i < len(lines) and not _HUNK_RE.match(lines[i]):
                ln = lines[i]
                if ln == "" and i == len(lines) - 1:
                    break
                tag, value = (ln[0], ln[1:]) if ln else (" ", "")
                if tag == "+":
                    prefix = f"{tgt:>4}|+"; tgt += 1; n_tgt += 1
                elif tag == "-":
                    prefix = f"{src:>4}|-"; src += 1; n_src += 1
                elif tag == " ":
                    prefix = f"{src:>4}| "; src += 1; tgt += 1; n_src += 1; n_tgt += 1
                elif tag == "\\":
                    i += 1
                    continue
                else:
                    raise ValueError(f"hunk diff line expected: {ln!r}")
                result.append(f"{prefix} {value.rstrip()}")
                i += 1
            if n_src != s_len or n_tgt != t_len:
                raise ValueError("hunk length mismatch")
        return "\n".join(result)
    except Exception:
        return None


def format_text_edit_output(path: str, before: str, after: str) -> str | None:
    diff = _compute_diff_unidiff(before.expandtabs(), after.expandtabs())
    if diff is None:
        return None
    return f"The file {path} has been edited. Here's a diff showing the changes made:\n{diff}"


def format_str_replace_action_output(arguments: dict[str, Any]) -> str | None:
    if arguments.get("command") != "str_replace":
        return None
    path = arguments.get("path")
    old_str = arguments.get("old_str")
    new_str = arguments.get("new_str") or ""
    if not isinstance(path, str) or not isinstance(old_str, str):
        return None
    return format_text_edit_output(path, old_str, new_str)
# fmt: on
# ── vendored from September: end ────────────────────────────────────────────


def load_template() -> str:
    """The September prompt template, hash-checked."""
    data = resources.files("whitebox_monitor.core").joinpath("data/old_action.md").read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != TEMPLATE_SHA256:
        raise RuntimeError(
            f"old_action.md was modified: sha256 {got} != {TEMPLATE_SHA256}"
        )
    return data.decode("utf-8")


def september_prompt(traj: dict, action_idx: int, template: str | None = None) -> MonitorPrompt:
    """Monitor prompt for ``traj["actions"][action_idx]``: one user message, no
    system prompt, exactly as September's ``build_messages``.

    Spans: ``"action"`` is the content of the ``<current_action>`` block and
    ``"history"`` the content of the ``<history>`` block, the same spans
    September used for ``action_end`` and ``transcript_end``.

    ``traj`` follows the September contract: ``main_task_description``,
    ``side_task_goals``, ``actions`` (dicts with ``function``, ``arguments``,
    ``output``, ``output_error``), optional ``environment_description``.
    """
    template = template if template is not None else load_template()
    desc = traj.get("main_task_description")
    if not desc:
        raise ValueError("Trajectory is missing a main task description.")
    goals = traj.get("side_task_goals")
    if goals is None:
        raise ValueError("Trajectory is missing side_task_goals")
    actions = traj["actions"]
    if not 0 <= action_idx < len(actions):
        raise IndexError(f"action_idx {action_idx} out of range")
    fields = {
        "conversation_history_minus_one": format_history(actions[:action_idx]),
        "current_action": format_action(actions[action_idx], hide_output=True),
        "task_description": desc,
        "environment_description": traj.get("environment_description") or "",
        "side_task_goals": goals,
    }
    text, spans = fill_template(
        template, fields,
        {"current_action": "action", "conversation_history_minus_one": "history"},
    )
    return MonitorPrompt(messages=[{"role": "user", "content": text}], spans=spans,
                         span_message_index=0)


def prompt_sha256(prompt: MonitorPrompt) -> str:
    """sha256 of the user text, September's ``prompt_sha256``."""
    return hashlib.sha256(prompt.content.encode()).hexdigest()


def load_side_task_goals(path: str | Path) -> dict[str, str]:
    return json.loads(Path(path).read_text())["goals_text"]


def load_export(data_dir: str | Path, goals_path: str | Path,
                sides: tuple[str, ...] = ("honest", "attack")) -> dict[str, dict]:
    """Read September's export (``trajectories.jsonl``, ``honest.jsonl``,
    ``attack.jsonl``) into trajectory dicts keyed by ``traj_id``."""
    data_dir = Path(data_dir)
    for name in ("trajectories.jsonl", *(f"{s}.jsonl" for s in sides)):
        if not (data_dir / name).is_file():
            raise FileNotFoundError(
                f"September export file missing: {data_dir / name}. The export "
                "lives on the GPU machine under "
                "checkpoints/local-monitor-activations/data/export."
            )
    goals = load_side_task_goals(goals_path)
    trajs: dict[str, dict] = {}
    with (data_dir / "trajectories.jsonl").open() as f:
        for line in f:
            t = json.loads(line)
            if t["side"] not in sides:
                continue
            t["side_task_goals"] = goals[t["env"]]
            t["actions"] = []
            trajs[t["traj_id"]] = t
    for side in sides:
        with (data_dir / f"{side}.jsonl").open() as f:
            for line in f:
                a = json.loads(line)
                t = trajs.get(a["traj_id"])
                if t is None:
                    continue
                t["actions"].append({
                    "index": a["action_idx"],
                    "function": a["tool_name"],
                    "arguments": a["tool_input"],
                    "output": a["tool_output"],
                    "output_error": a["output_error"],
                })
    for t in trajs.values():
        t["actions"].sort(key=lambda a: a["index"])
        if [a["index"] for a in t["actions"]] != list(range(len(t["actions"]))):
            raise ValueError(f"trajectory {t['traj_id']} has gaps in its actions")
    return trajs
