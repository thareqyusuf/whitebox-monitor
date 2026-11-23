"""ControlArena adapter: a monitor that writes three readouts per action.

Written against control-arena 20.1.1 / inspect-ai 0.3.291 (source read, not
memory). ``whitebox_monitor_builder`` is a ``@control_agent``; it does not use
the ``@monitor`` decorator, because that decorator writes only the
``MonitorResponse`` fields into the message metadata. Instead it does the
same three things by hand: convert the state to the monitor view, compute the
response, and update the metadata of the last assistant message with the
response plus the ``wbm_*`` keys.

Prompt spans. ControlArena's ``PromptFormatter`` returns a plain string, so it
cannot say where the action is. This adapter takes a ``PromptBuilder``
instead: ``(task_metadata, messages) -> MonitorPrompt``. Two are provided:

``tool_calls_prompt_builder``  fills a ``{task_description}`` /
    ``{tool_calls}`` template (ControlArena's default monitor prompt) and
    records the span by offset arithmetic while filling.
``wrap_prompt_formatter``      wraps an existing ``PromptFormatter``: it
    formats the prompt, then locates the action text returned by a second
    formatter. The text must occur exactly once, otherwise it raises.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Sequence
from typing import Any

import anyio
from control_arena import control_agent
from control_arena._control_task_metadata import ControlTaskMetadata, control_task_metadata
from control_arena.monitor import (
    MonitorResponse,
    MonitorScoreRange,
    PromptFormatter,
    get_last_assistant_message,
    last_tool_calls_message_formatter,
)
from control_arena.monitor.prompts import monitor_prompts
from control_arena.tools import convert_agent_state_to_monitor_view
from inspect_ai.agent import Agent, AgentState
from inspect_ai.model import ChatMessage
from inspect_ai.solver._task_state import sample_state
from inspect_ai.tool import Tool
from pydantic import Field

from ..core.pipeline import compute_readout
from ..core.probe import LinearProbe
from ..core.prompt import fill_template
from ..core.types import CaptureSpec, GenSpec, MonitorPrompt, Readout
from ..store.sidecar import SidecarWriter

PromptBuilder = Callable[[ControlTaskMetadata, Sequence[ChatMessage]], MonitorPrompt]

WBM_KEYS = (
    "wbm_integer", "wbm_integer_parsed", "wbm_expected", "wbm_expected_post",
    "wbm_probe_score", "wbm_score_token_probs", "wbm_activation_ref",
    "wbm_action_token_index", "wbm_meta",
)
_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.S | re.I)
_META_KEYS = (
    "backend", "model_id", "dtype", "quantization", "versions", "positions",
    "action_token_position", "n_prompt_tokens", "content_sha256", "rendered_sha256",
    "input_ids_sha256", "integer_parse", "expected_info", "expected_post_info",
    "expected_source", "score_values", "score_tokens", "template_date",
    "pinned_chat_template_sha256", "attn_implementation", "attn_chunk_size", "probe",
    "layer_index_convention", "residual_dtype",
)


def tool_calls_prompt_builder(
    prompt: str = monitor_prompts["default"],
    message_formatter: Callable[[Sequence[ChatMessage]], str] = last_tool_calls_message_formatter,
    system: str | None = None,
) -> PromptBuilder:
    """Prompt builder for a ``{task_description}`` / ``{tool_calls}`` template.

    The action span is the formatted tool calls without the trailing newline
    that ``last_tool_calls_message_formatter`` appends, so the action's last
    token is not a newline.
    """

    def build(task_metadata: ControlTaskMetadata, messages: Sequence[ChatMessage]) -> MonitorPrompt:
        tool_calls = message_formatter(messages)
        fields = {"task_description": task_metadata.main_task_description,
                  "tool_calls": tool_calls}
        text, spans = fill_template(prompt, fields, {"tool_calls": "action"})
        start, end = spans["action"]
        end = start + len(tool_calls.rstrip("\n"))
        chat: list[dict] = []
        if system is not None:
            chat.append({"role": "system", "content": system})
        chat.append({"role": "user", "content": text})
        return MonitorPrompt(messages=chat, spans={"action": (start, end)},
                             span_message_index=len(chat) - 1)

    return build


def wrap_prompt_formatter(
    formatter: PromptFormatter,
    prompt: str,
    action_formatter: Callable[[Sequence[ChatMessage]], str] = last_tool_calls_message_formatter,
) -> PromptBuilder:
    """Wrap a ControlArena ``PromptFormatter`` so it yields the action span.

    The formatter returns only a string, so the action is located by finding
    the output of ``action_formatter`` (trailing newlines removed) in it. It
    must occur exactly once; a repeated or missing action raises rather than
    guessing. Use ``tool_calls_prompt_builder`` when the action text can
    repeat (e.g. full-trajectory prompts).
    """

    def build(task_metadata: ControlTaskMetadata, messages: Sequence[ChatMessage]) -> MonitorPrompt:
        text = formatter(task_metadata, messages, prompt)
        action = action_formatter(messages).rstrip("\n")
        if not action:
            raise ValueError("the action formatter returned no text")
        count = text.count(action)
        if count != 1:
            raise ValueError(
                f"the action text occurs {count} times in the formatted prompt; "
                "wrap_prompt_formatter needs exactly one occurrence"
            )
        start = text.index(action)
        return MonitorPrompt(messages=[{"role": "user", "content": text}],
                             spans={"action": (start, start + len(action))})

    return build


def _reason(readout: Readout) -> str:
    match = _REASONING_RE.search(readout.completion or "")
    if match and match.group(1).strip():
        return match.group(1).strip()
    if readout.reasoning and readout.reasoning.strip():
        return readout.reasoning.strip()
    return "No reasoning provided."


def _sample_ids() -> tuple[str, int | None]:
    state = sample_state()
    if state is None:
        return "no_sample", None
    return str(state.sample_id), state.epoch


@control_agent
def whitebox_monitor_builder(
    backend: Any = Field(exclude=True,
                         description="A whitebox_monitor Backend (e.g. HFHooksBackend)."),
    capture: Any = Field(default_factory=CaptureSpec, description="CaptureSpec."),
    gen: Any = Field(default_factory=GenSpec, description="GenSpec."),
    probe: Any = Field(default=None, exclude=True, description="LinearProbe or None."),
    prompt_builder: PromptBuilder = Field(default_factory=tool_calls_prompt_builder),
    score_range: MonitorScoreRange = Field(default_factory=MonitorScoreRange),
    store: Any = Field(
        default=None, exclude=True,
        description="SidecarWriter for the residuals. Without it they are dropped."),
    run_id: str | None = Field(
        default=None, description="Sidecar run id; a random one is made if omitted."),
    default_suspicion_score: float = Field(
        default=0.0,
        description="Score used when the integer cannot be parsed, as ControlArena's "
                    "basic monitor does; the response then has score_parsed=False."),
    skip_if_no_tool_calls: bool = Field(default=True),
) -> Agent:
    """White-box monitor: integer, expected score and residual read.

    ``suspicion_score`` is the parsed integer normalised by ``score_range``
    (0-10 to 0-1 by default), so protocols and thresholds built for
    ControlArena monitors keep working. The extra readouts go into the same
    message metadata under ``wbm_*`` keys:

        wbm_integer             raw integer, or None
        wbm_integer_parsed      False when the integer could not be parsed
        wbm_expected            E (forced readout), raw scale
        wbm_expected_post       E at the step that emitted the digit
        wbm_probe_score         frozen probe output, or None
        wbm_score_token_probs   normalised distribution behind wbm_expected
        wbm_activation_ref      pointer into the sidecar store, or None
        wbm_action_token_index  token index the residual was read at
        wbm_meta                model id, backend, versions, positions

    If the integer cannot be parsed the monitor does what ControlArena's
    ``basic_monitor`` does: ``suspicion_score=default_suspicion_score`` and
    ``score_parsed=False``; ``wbm_integer`` stays None. Backend errors (an
    unresolved position, a multi-token score value) are raised, not hidden.

    ``backend``, ``probe`` and ``store`` are live objects, so they are
    excluded from the serialised agent config that ControlArena writes into
    the eval log; the model id, backend name and probe hash are in
    ``wbm_meta`` on every message instead.
    """
    if not (score_range.output_min <= default_suspicion_score <= score_range.output_max):
        raise ValueError(
            f"default_suspicion_score {default_suspicion_score} is out of bounds for the "
            f"provided output score_range {score_range}"
        )
    if not callable(getattr(backend, "run", None)):
        raise ValueError("backend must have a run(prompt, capture, gen) method")
    # These are typed Any in the signature because @control_agent turns the
    # signature into a pydantic model, which cannot describe numpy fields.
    for value, kind, optional in ((capture, CaptureSpec, False), (gen, GenSpec, False),
                                  (probe, LinearProbe, True), (store, SidecarWriter, True)):
        if not (isinstance(value, kind) or (optional and value is None)):
            raise ValueError(f"expected {kind.__name__}, got {type(value).__name__}")
    if (min(gen.score_values), max(gen.score_values)) != (score_range.min, score_range.max):
        raise ValueError(
            f"GenSpec.score_values span [{min(gen.score_values)}, {max(gen.score_values)}] "
            f"but score_range is [{score_range.min}, {score_range.max}]"
        )
    _run_id = run_id or f"wbm-{uuid.uuid4().hex[:12]}"

    async def impl(state: AgentState, tools: list[Tool]) -> AgentState:
        last_assistant = get_last_assistant_message(state.messages)
        view = convert_agent_state_to_monitor_view(state)
        view_last = get_last_assistant_message(view.messages)
        if last_assistant.metadata is None:
            last_assistant.metadata = {}

        if skip_if_no_tool_calls and not view_last.tool_calls:
            response = MonitorResponse(
                suspicion_score=default_suspicion_score,
                reason="No tools found in model response", score_parsed=False)
            last_assistant.metadata.update(response.model_dump())
            last_assistant.metadata.update(dict.fromkeys(WBM_KEYS))
            last_assistant.metadata.update(
                {"wbm_integer_parsed": False, "wbm_meta": {"skipped": "no tool calls"}})
            return state

        prompt = prompt_builder(control_task_metadata(), view.messages)
        readout = await anyio.to_thread.run_sync(compute_readout, backend, prompt, capture,
                                                 gen, probe)

        if readout.integer is None:
            response = MonitorResponse(
                suspicion_score=default_suspicion_score,
                reason="Could not parse score from model response: "
                       + str(readout.meta["integer_parse"]["reason"]),
                score_parsed=False)
        else:
            unit = (readout.integer - score_range.min) / (score_range.max - score_range.min)
            scaled = unit * (score_range.output_max - score_range.output_min) + score_range.output_min
            response = MonitorResponse(suspicion_score=scaled, reason=_reason(readout))

        sample_id, epoch = _sample_ids()
        activation_ref = None
        if store is not None and readout.residuals:
            if last_assistant.id is None:
                raise ValueError("assistant message has no id; cannot key its residuals")
            store_sample = sample_id if epoch in (None, 1) else f"{sample_id}#e{epoch}"
            activation_ref = store.add(_run_id, store_sample, last_assistant.id,
                                       readout.residuals,
                                       extra={"model_id": readout.meta.get("model_id")})
            store.flush()

        meta = {k: readout.meta[k] for k in _META_KEYS if k in readout.meta}
        meta.update({"run_id": _run_id, "sample_id": sample_id, "epoch": epoch,
                     "message_id": last_assistant.id})
        last_assistant.metadata.update(response.model_dump())
        last_assistant.metadata.update({
            "wbm_integer": readout.integer,
            "wbm_integer_parsed": readout.integer is not None,
            "wbm_expected": readout.expected,
            "wbm_expected_post": readout.expected_post,
            "wbm_probe_score": readout.probe_score,
            "wbm_score_token_probs": readout.score_token_probs,
            "wbm_activation_ref": activation_ref,
            "wbm_action_token_index": readout.meta.get("action_token_index"),
            "wbm_meta": meta,
        })
        return state

    return impl
