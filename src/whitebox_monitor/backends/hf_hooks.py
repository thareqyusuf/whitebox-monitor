"""Reference backend: Hugging Face Transformers with forward hooks.

Follows September's ``run_monitor.py`` step for step:

1. render with the chat template, tokenize the rendered string, resolve the
   capture positions;
2. ONE prefill forward pass over the prompt with forward hooks on the
   embedding module and every decoder block; the residual at the requested
   prompt positions is kept (layer 0 = embedding output, layer i = output of
   block i - 1);
3. forced readout on a deep copy of the prefill cache: each segment of
   ``GenSpec.forced_prefix`` is fed in its own forward call, and the score
   token probabilities are read after the last one (September's E);
4. greedy unforced generation on the original cache; the score is parsed
   from the output and the distribution of the step that emitted the digit
   is September's ``expected_score_post``.

Everything happens on one set of ``input_ids``, so the residuals and both
distributions come from the same prompt bytes. Batch size 1. Greedy only.

Only prompt-side positions can be captured. September's generation-side
positions (``analysis_end``, ``reasoning_end``, ``pre_score``) are not
implemented and raise.

``HFHooksBackend.gpt_oss_20b()`` reproduces September's model settings
(MXFP4, bf16, ``chunked_eager`` attention with a fixed chunk size, harmony
forced prefix and stop tokens, pinned template date). That preset needs a GPU
and has not been run from this package unless the README says so.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..core.positions import PositionError, render_and_tokenize, resolve_positions
from ..core.readouts import parse_integer, score_token_ids
from ..core.types import CaptureSpec, GenSpec, MonitorPrompt, RawOutput

GENERATION_POSITIONS = ("analysis_end", "reasoning_end", "pre_score")

HARMONY_FINAL = "<|channel|>final<|message|>"
HARMONY_ANALYSIS = "<|channel|>analysis<|message|>"
HARMONY_FORCED_PREFIX = (HARMONY_FINAL, "<score>\n")
GPT_OSS_STOP_IDS = (200002, 200012)  # <|return|>, <|call|> (not <|end|>)
GPT_OSS_FINAL_PROMPT_TOKEN_ID = 173781  # "assistant"


@dataclass(frozen=True)
class HFConfig:
    model_id: str
    dtype: str = "float32"
    device: str = "cpu"
    attn_implementation: str | None = None
    #: fixed query-chunk size for ``chunked_eager`` (September: 256)
    attn_chunk_size: int | None = None
    #: extra kwargs for ``apply_chat_template`` (e.g. reasoning_effort)
    template_kwargs: Mapping[str, Any] = field(default_factory=dict)
    #: replace the template's ``strftime_now("%Y-%m-%d")`` with this literal
    template_date: str | None = None
    #: "harmony" | "think_tags" | "none"
    output_format: str = "none"
    #: extra stop token ids for the unforced generation (EOS is always a stop)
    stop_token_ids: tuple[int, ...] = ()
    #: pass cache_position / position_ids explicitly (needed by chunked_eager)
    pass_positions: bool = False
    #: dotted paths to the embedding module and the list of decoder blocks
    embed_path: str = "model.embed_tokens"
    layers_path: str = "model.layers"
    #: refuse to run if the weights were dequantized (September asserted MXFP4)
    require_native_quantization: bool = False
    #: assert the last prompt token id (September: 173781 for gpt-oss)
    expect_final_token_id: int | None = None
    max_prompt_tokens: int | None = None
    #: dtype of the returned residuals
    residual_dtype: str = "float16"


def _resolve_attr(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def pin_chat_template(tokenizer: Any, template_date: str) -> str:
    """September's ``pin_chat_template``: make the date in the chat template a
    literal. Returns the sha256 of the pinned template. Idempotent."""
    template = tokenizer.chat_template
    needle = 'strftime_now("%Y-%m-%d")'
    literal = f'"{template_date}"'
    if needle in template:
        if template.count(needle) != 1:
            raise ValueError("chat template has more than one date call")
        tokenizer.chat_template = template.replace(needle, literal)
    if literal not in tokenizer.chat_template or "strftime_now" in tokenizer.chat_template:
        raise ValueError("could not pin the date in the chat template")
    return hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()


def split_output(text: str, output_format: str) -> tuple[str | None, str | None, dict[str, Any]]:
    """Split generated text into ``(final_text, reasoning, info)``.

    harmony     final = text after ``<|channel|>final<|message|>`` (September
                parses the score only there); reasoning = analysis channel.
    think_tags  reasoning = ``<think>...</think>``; final = the rest.
    none        final = whole text, reasoning = None.
    """
    info: dict[str, Any] = {"output_format": output_format, "final_offset": None}
    if output_format == "harmony":
        reasoning = None
        if HARMONY_ANALYSIS in text:
            reasoning = text.split(HARMONY_ANALYSIS, 1)[1].split("<|end|>", 1)[0]
        info["opened_analysis"] = HARMONY_ANALYSIS in text
        info["has_final_channel"] = HARMONY_FINAL in text
        if HARMONY_FINAL not in text:
            info["final_text_reason"] = "no final channel in the output"
            return None, reasoning, info
        offset = text.index(HARMONY_FINAL) + len(HARMONY_FINAL)
        info["final_offset"] = offset
        return text[offset:], reasoning, info
    if output_format == "think_tags":
        if "</think>" in text:
            head, tail = text.split("</think>", 1)
            info["final_offset"] = len(head) + len("</think>")
            return tail, head.split("<think>", 1)[-1], info
        if "<think>" in text:
            info["final_text_reason"] = "unterminated <think> block"
            return None, text.split("<think>", 1)[1], info
        info["final_offset"] = 0
        return text, None, info
    if output_format == "none":
        info["final_offset"] = 0
        return text, None, info
    raise ValueError(f"unknown output_format {output_format!r}")


class _Capture:
    """Forward hooks on the embedding module and every decoder block. Armed
    only during the prefill pass; keeps ``output[0, positions, :]``."""

    def __init__(self, embed: torch.nn.Module, layers: Sequence[torch.nn.Module]) -> None:
        self.n = len(layers) + 1
        self.buf: list[torch.Tensor | None] = [None] * self.n
        self.positions: torch.Tensor | None = None
        self.wanted: set[int] = set()
        self.handles = [embed.register_forward_hook(self._hook(0))] + [
            layer.register_forward_hook(self._hook(i + 1)) for i, layer in enumerate(layers)
        ]

    def _hook(self, i: int):
        def fn(module, inputs, output):
            if self.positions is None or i not in self.wanted:
                return
            out = output[0] if isinstance(output, (tuple, list)) else output
            if out.dim() != 3:
                return
            self.buf[i] = out[0, self.positions.to(out.device), :].detach().float().cpu()
        return fn

    def arm(self, layers: Sequence[int], positions: Sequence[int]) -> None:
        self.buf = [None] * self.n
        self.wanted = set(layers)
        self.positions = torch.tensor(list(positions), dtype=torch.long)

    def take(self) -> dict[int, torch.Tensor]:
        self.positions = None
        missing = [i for i in self.wanted if self.buf[i] is None]
        if missing:
            raise RuntimeError(f"layers {missing} did not fire during prefill")
        out = {i: self.buf[i] for i in self.wanted}
        self.buf = [None] * self.n
        return out  # type: ignore[return-value]

    def disarm(self) -> None:
        self.positions = None


class HFHooksBackend:
    """See module docstring. One pass for residuals, forced E and generation."""

    name = "hf_hooks"

    def __init__(self, config: HFConfig, model: Any = None, tokenizer: Any = None) -> None:
        self.config = config
        self._lock = threading.Lock()
        self.load_seconds: float | None = None
        if (model is None) != (tokenizer is None):
            raise ValueError("pass both model and tokenizer, or neither")
        if model is None:
            model, tokenizer = self._load()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.template_sha256: str | None = None
        if config.template_date is not None:
            self.template_sha256 = pin_chat_template(tokenizer, config.template_date)
        elif tokenizer.chat_template and "strftime_now" in tokenizer.chat_template:
            raise ValueError(
                "the chat template inserts today's date; set HFConfig.template_date "
                "so prompts are reproducible"
            )
        self.embed = _resolve_attr(self.model, config.embed_path)
        self.layers = list(_resolve_attr(self.model, config.layers_path))
        self.n_layers = len(self.layers)
        self._capture = _Capture(self.embed, self.layers)
        params = inspect.signature(self.model.forward).parameters
        self._logits_to_keep = "logits_to_keep" in params
        stops = set(config.stop_token_ids)
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            stops.add(int(eos))
        if not stops:
            raise ValueError("no stop token: tokenizer has no EOS and none were given")
        self.stop_ids = stops
        self._score_ids_cache: dict[tuple, tuple[dict[int, int], dict[str, Any]]] = {}

    # ── construction ───────────────────────────────────────────────────────
    def _load(self) -> tuple[Any, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self.config
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        kwargs: dict[str, Any] = {"dtype": getattr(torch, cfg.dtype)}
        if cfg.attn_implementation == "chunked_eager":
            from . import _chunked_attention

            _chunked_attention.CHUNK_SIZE = cfg.attn_chunk_size
            _chunked_attention.register()
        if cfg.attn_implementation is not None:
            kwargs["attn_implementation"] = cfg.attn_implementation
        if cfg.device != "cpu":
            kwargs["device_map"] = {"": cfg.device}
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
        self.load_seconds = time.time() - t0
        if cfg.attn_implementation is not None:
            got = getattr(model.config, "_attn_implementation", None)
            if got != cfg.attn_implementation:
                raise RuntimeError(f"attention implementation is {got!r}, "
                                   f"wanted {cfg.attn_implementation!r}")
        if cfg.require_native_quantization:
            quantizer = getattr(model, "hf_quantizer", None)
            if quantizer is None or quantizer.quantization_config.dequantize:
                raise RuntimeError(
                    "weights were dequantized (or carry no quantizer); September "
                    "ran native MXFP4. Install `kernels` and use a GPU with "
                    "MXFP4 kernels, or set require_native_quantization=False."
                )
        return model, tokenizer

    @classmethod
    def gpt_oss_20b(cls, *, template_date: str, reasoning_effort: str = "low",
                    attn_chunk_size: int = 256, device: str = "cuda:0",
                    max_prompt_tokens: int = 32000,
                    require_native_quantization: bool = True) -> HFHooksBackend:
        """September's gpt-oss-20b settings. ``template_date`` must be the date
        the cache was produced with (2026-09-10 for the main pass)."""
        return cls(HFConfig(
            model_id="openai/gpt-oss-20b", dtype="bfloat16", device=device,
            attn_implementation="chunked_eager", attn_chunk_size=attn_chunk_size,
            template_kwargs={"reasoning_effort": reasoning_effort},
            template_date=template_date, output_format="harmony",
            stop_token_ids=GPT_OSS_STOP_IDS, pass_positions=True,
            require_native_quantization=require_native_quantization,
            expect_final_token_id=GPT_OSS_FINAL_PROMPT_TOKEN_ID,
            max_prompt_tokens=max_prompt_tokens,
        ))

    @staticmethod
    def gpt_oss_genspec(max_new_tokens: int = 1024) -> GenSpec:
        return GenSpec(max_new_tokens=max_new_tokens, forced_prefix=HARMONY_FORCED_PREFIX)

    # ── helpers ────────────────────────────────────────────────────────────
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _forward(self, ids: Sequence[int], past: Any, start: int) -> Any:
        device = self.device
        kwargs: dict[str, Any] = {"use_cache": True}
        if past is not None:
            kwargs["past_key_values"] = past
        if self._logits_to_keep:
            kwargs["logits_to_keep"] = 1
        if self.config.pass_positions and past is not None:
            cp = torch.arange(start, start + len(ids), device=device)
            kwargs["cache_position"] = cp
            kwargs["position_ids"] = cp[None]
        return self.model(torch.tensor([list(ids)], device=device), **kwargs)

    def _score_ids(self, gen: GenSpec) -> tuple[dict[int, int], dict[str, Any]]:
        key = (gen.score_values, gen.multi_token_scores)
        if key not in self._score_ids_cache:
            self._score_ids_cache[key] = score_token_ids(
                self.tokenizer, gen.score_values, gen.multi_token_scores)
        return self._score_ids_cache[key]

    @staticmethod
    def _score_probs(logits: torch.Tensor, ids: Mapping[int, int]) -> dict[int, float]:
        probs = torch.softmax(logits.float(), dim=-1)  # float32, as September
        return {value: float(probs[token]) for value, token in ids.items()}

    def base_meta(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "backend": self.name, "model_id": cfg.model_id, "dtype": cfg.dtype,
            "device": str(self.device), "attn_implementation": cfg.attn_implementation,
            "attn_chunk_size": cfg.attn_chunk_size, "softmax_dtype": "float32",
            "template_kwargs": dict(cfg.template_kwargs),
            "template_date": cfg.template_date,
            "pinned_chat_template_sha256": self.template_sha256,
            "output_format": cfg.output_format, "stop_token_ids": sorted(self.stop_ids),
            "n_layers": self.n_layers,
            "layer_index_convention": "0 = embedding output, i = output of decoder block i-1",
            "residual_dtype": cfg.residual_dtype,
            "quantization": self._quantization(),
        }

    def _quantization(self) -> str | None:
        quantizer = getattr(self.model, "hf_quantizer", None)
        if quantizer is None:
            return None
        qc = quantizer.quantization_config
        return f"{type(qc).__name__}(dequantize={getattr(qc, 'dequantize', None)})"

    # ── the run ────────────────────────────────────────────────────────────
    def run(self, prompt: MonitorPrompt, capture: CaptureSpec, gen: GenSpec) -> RawOutput:
        with self._lock, torch.no_grad():
            return self._run(prompt, capture, gen)

    def _run(self, prompt: MonitorPrompt, capture: CaptureSpec, gen: GenSpec) -> RawOutput:
        cfg, tok = self.config, self.tokenizer
        for name in capture.positions:
            if name in GENERATION_POSITIONS:
                raise PositionError(
                    f"{name!r} is a generation-side position; this backend "
                    "captures prompt-side positions only"
                )
        bad = [layer for layer in capture.layers if layer > self.n_layers]
        if bad:
            raise ValueError(f"layers {bad} do not exist (valid: 0..{self.n_layers})")

        t0 = time.time()
        tokenized = render_and_tokenize(tok, prompt, cfg.template_kwargs)
        ids = tokenized.input_ids
        if cfg.expect_final_token_id is not None and ids[-1] != cfg.expect_final_token_id:
            raise RuntimeError(f"last prompt token is {ids[-1]}, expected "
                               f"{cfg.expect_final_token_id}")
        if cfg.max_prompt_tokens is not None and len(ids) > cfg.max_prompt_tokens:
            raise ValueError(f"prompt is {len(ids)} tokens, over max_prompt_tokens "
                             f"{cfg.max_prompt_tokens}")
        positions = resolve_positions(capture.positions, tokenized, prompt, tok)
        score_ids, score_info = self._score_ids(gen)
        forced_ids = []
        for segment in gen.forced_prefix:
            seg_ids = tok.encode(segment, add_special_tokens=False)
            if tok.decode(seg_ids) != segment:
                raise ValueError(f"forced segment {segment!r} does not round-trip "
                                 "through the tokenizer")
            forced_ids.append(seg_ids)

        meta = self.base_meta()
        meta.update({
            "n_prompt_tokens": len(ids),
            "rendered_sha256": hashlib.sha256(tokenized.rendered.encode()).hexdigest(),
            "content_sha256": hashlib.sha256(prompt.content.encode()).hexdigest(),
            "input_ids_sha256": hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest(),
            "score_token_ids": {str(k): v for k, v in score_ids.items()},
            "score_tokens": score_info,
            "forced_prefix": list(gen.forced_prefix), "forced_prefix_ids": forced_ids,
            "max_new_tokens": gen.max_new_tokens,
        })

        # 2. prefill with capture
        order = list(capture.positions)
        if capture.layers and order:
            self._capture.arm(capture.layers, [positions[n].token_index for n in order])
        try:
            t1 = time.time()
            out = self._forward(ids, None, 0)
            meta["prefill_seconds"] = time.time() - t1
            captured = self._capture.take() if capture.layers and order else {}
        finally:
            self._capture.disarm()
        residuals: dict[tuple[int, str], np.ndarray] = {}
        dtype = np.dtype(cfg.residual_dtype)
        for layer, rows in captured.items():
            for j, name in enumerate(order):
                vec = rows[j].numpy()
                if not np.all(np.isfinite(vec)):
                    raise RuntimeError(f"non-finite residual at layer {layer}, {name}")
                residuals[(layer, name)] = vec.astype(dtype)
        past = out.past_key_values
        last_logits = out.logits[0, -1].float()
        n_prompt = len(ids)

        # 3. forced readout on a throwaway copy of the cache
        forced_probs: dict[int, float] | None = None
        if forced_ids:
            past_copy = copy.deepcopy(past)
            pos = n_prompt
            o = None
            for seg_ids in forced_ids:
                o = self._forward(seg_ids, past_copy, pos)
                past_copy = o.past_key_values
                pos += len(seg_ids)
            assert o is not None
            forced_probs = self._score_probs(o.logits[0, -1], score_ids)
            del past_copy, o
        else:
            meta["forced_reason"] = "GenSpec.forced_prefix is empty"

        # 4. greedy unforced generation
        if not gen.generate:
            meta.update({"post_reason": "generation skipped (GenSpec.generate=False)",
                         "final_text_reason": "generation skipped (GenSpec.generate=False)",
                         "seconds": time.time() - t0})
            return RawOutput("", None, None, forced_probs, None, residuals, positions, meta)

        t2 = time.time()
        gen_ids: list[int] = []
        step_probs: list[dict[int, float]] = []
        logits = last_logits
        pos = n_prompt
        for _ in range(gen.max_new_tokens):
            step_probs.append(self._score_probs(logits, score_ids))
            nxt = int(logits.argmax())
            gen_ids.append(nxt)
            if nxt in self.stop_ids:
                break
            o = self._forward([nxt], past, pos)
            past = o.past_key_values
            logits = o.logits[0, -1].float()
            pos += 1
        text = tok.decode(gen_ids, skip_special_tokens=False)
        final_text, reasoning, split_info = split_output(text, cfg.output_format)
        meta.update(split_info)
        meta.update({
            "generation_seconds": time.time() - t2, "n_generated": len(gen_ids),
            "hit_max_new_tokens": bool(gen_ids) and gen_ids[-1] not in self.stop_ids
            and len(gen_ids) >= gen.max_new_tokens,
            "stopped_on": tok.decode([gen_ids[-1]]) if gen_ids and gen_ids[-1] in self.stop_ids else None,
            "generated_ids": gen_ids,
        })

        # distribution of the step that emitted the parsed digit
        post_probs: dict[int, float] | None = None
        lo, hi = min(gen.score_values), max(gen.score_values)
        integer, parse_info = parse_integer(final_text, lo, hi)
        if integer is None:
            meta["post_reason"] = "no parsed score: " + str(parse_info["reason"])
        else:
            char = split_info["final_offset"] + parse_info["match_span"][0]
            ends = [len(tok.decode(gen_ids[: i + 1], skip_special_tokens=False))
                    for i in range(len(gen_ids))]
            starts = [0] + ends[:-1]
            step = next((i for i in range(len(gen_ids)) if starts[i] <= char < ends[i]), None)
            if step is None:
                meta["post_reason"] = "could not map the digit to a generation step"
            else:
                digit_token = tok.decode([gen_ids[step]])
                clean = starts[step] == char and digit_token == str(parse_info["raw"])
                meta["pre_score"] = {"digit_gen_index": step, "clean": clean,
                                     "digit_token": digit_token}
                if clean:
                    post_probs = step_probs[step]
                else:
                    meta["post_reason"] = (f"digit token is {digit_token!r}, parsed "
                                           f"{parse_info['raw']}: not a clean score step")
        meta["seconds"] = time.time() - t0
        return RawOutput(text, final_text, reasoning, forced_probs, post_probs,
                         residuals, positions, meta)
