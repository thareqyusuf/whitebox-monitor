"""vLLM-Lens backend (phase 5). NOT RUN against a server yet.

Written from the vLLM-Lens v1.3.0 source (commit 3d11fb0), not from a live
server; ``docs/backend-notes.md`` lists what was read and what is unverified.
The request/response handling is unit-tested against a fake transport only.

It talks to the raw OpenAI-compatible ``/v1/completions`` endpoint with the
prompt as a list of token ids, which is vLLM-Lens's tested path and avoids
the server re-rendering the chat template. The ids are the ones
``core.positions`` resolved the capture positions on.

THREE passes, all over the same prompt token ids:

  A  prompt, ``max_tokens=1``, ``vllm_xargs.output_residual_stream`` ->
     residuals. vLLM-Lens cannot restrict capture to positions, so the whole
     ``(n_layers, n_positions, d)`` tensor comes back and is sliced here.
     Keep ``CaptureSpec.layers`` short for long prompts.
  B  prompt + forced prefix ids, ``max_tokens=1``, ``logprobs=k`` -> E.
  C  prompt, greedy generation with ``logprobs=k`` -> integer and E_post.

Differences from ``hf_hooks`` that the parity test has to measure:

* E comes from the top-k logprobs, not the full softmax. Score tokens outside
  the top-k count as 0 and are listed in ``meta["forced_logprobs"]["missing"]``.
* vLLM-Lens layer ``i`` is the output of decoder block ``i``. This package's
  layer ``i`` is block ``i - 1`` and layer 0 is the embedding output, so layer
  ``L`` is requested as ``L - 1``; layer 0 is not available and raises.
* The forced prefix is fed in the same request as the prompt, not as separate
  cached calls.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.positions import PositionError, render_and_tokenize, resolve_positions
from ..core.readouts import parse_integer, probs_from_logprobs, score_token_ids
from ..core.types import CaptureSpec, GenSpec, MonitorPrompt, RawOutput
from .hf_hooks import GENERATION_POSITIONS, split_output

Transport = Callable[[str, dict], dict]


def _http_post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.loads(response.read())


def _decode_activations(entry: Any) -> np.ndarray:
    """Decode ``resp["activations"]["residual_stream"]`` with vLLM-Lens's own
    deserializer (base64 + zstd, bf16 shipped as an int16 view)."""
    if isinstance(entry, np.ndarray):
        return entry
    from vllm_lens import deserialize_tensor  # type: ignore[import-not-found]

    tensor = deserialize_tensor(entry)
    return tensor.float().cpu().numpy() if hasattr(tensor, "float") else np.asarray(tensor)


@dataclass(frozen=True)
class VLLMLensConfig:
    base_url: str  # e.g. http://localhost:8000
    model_id: str
    template_kwargs: Mapping[str, Any] = field(default_factory=dict)
    output_format: str = "none"
    top_logprobs: int = 20
    residual_dtype: str = "float16"
    max_prompt_tokens: int | None = None


class VLLMLensBackend:
    """See module docstring: three passes, byte-identical prompt ids."""

    name = "vllm_lens"

    def __init__(self, config: VLLMLensConfig, tokenizer: Any,
                 transport: Transport | None = None,
                 decode_activations: Callable[[Any], np.ndarray] = _decode_activations) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self._post = transport or _http_post
        self._decode = decode_activations
        if tokenizer.chat_template and "strftime_now" in tokenizer.chat_template:
            raise ValueError("pin the chat template date first "
                             "(whitebox_monitor.backends.hf_hooks.pin_chat_template)")

    def base_meta(self) -> dict[str, Any]:
        cfg = self.config
        return {"backend": self.name, "model_id": cfg.model_id, "base_url": cfg.base_url,
                "template_kwargs": dict(cfg.template_kwargs), "output_format": cfg.output_format,
                "top_logprobs": cfg.top_logprobs, "residual_dtype": cfg.residual_dtype,
                "layer_index_convention": "0 = embedding output (unavailable), "
                                          "i = output of decoder block i-1",
                "passes": 3}

    def _complete(self, prompt_ids: list[int], **params: Any) -> dict:
        payload = {"model": self.config.model_id, "prompt": prompt_ids, "temperature": 0.0,
                   "skip_special_tokens": False, **params}
        return self._post(self.config.base_url.rstrip("/") + "/v1/completions", payload)

    def run(self, prompt: MonitorPrompt, capture: CaptureSpec, gen: GenSpec) -> RawOutput:
        cfg, tok = self.config, self.tokenizer
        for name in capture.positions:
            if name in GENERATION_POSITIONS:
                raise PositionError(f"{name!r} is a generation-side position; not supported")
        if 0 in capture.layers:
            raise ValueError("layer 0 (embedding output) is not available through vLLM-Lens "
                             "residual capture; request layers >= 1")
        tokenized = render_and_tokenize(tok, prompt, cfg.template_kwargs)
        ids = tokenized.input_ids
        if cfg.max_prompt_tokens is not None and len(ids) > cfg.max_prompt_tokens:
            raise ValueError(f"prompt is {len(ids)} tokens, over {cfg.max_prompt_tokens}")
        positions = resolve_positions(capture.positions, tokenized, prompt, tok)
        score_ids, score_info = score_token_ids(tok, gen.score_values, gen.multi_token_scores)
        values = sorted(score_ids)
        meta = self.base_meta()
        meta.update({
            "n_prompt_tokens": len(ids),
            "rendered_sha256": hashlib.sha256(tokenized.rendered.encode()).hexdigest(),
            "content_sha256": hashlib.sha256(prompt.content.encode()).hexdigest(),
            "input_ids_sha256": hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest(),
            "score_tokens": score_info, "forced_prefix": list(gen.forced_prefix),
        })

        # A: residuals
        residuals: dict[tuple[int, str], np.ndarray] = {}
        if capture.layers and capture.positions:
            layers = sorted(capture.layers)
            resp = self._complete(ids, max_tokens=1, vllm_xargs={
                "output_residual_stream": json.dumps([layer - 1 for layer in layers])})
            if "activations" not in resp or "residual_stream" not in resp["activations"]:
                raise RuntimeError("the server returned no residual_stream activations; is the "
                                   "vllm-lens plugin loaded?")
            acts = self._decode(resp["activations"]["residual_stream"])
            if acts.ndim != 3 or acts.shape[0] != len(layers):
                raise RuntimeError(f"unexpected activation shape {acts.shape} for {len(layers)} layers")
            if acts.shape[1] < len(ids):
                raise RuntimeError(f"activations cover {acts.shape[1]} positions, prompt has "
                                   f"{len(ids)} tokens")
            meta["activation_shape"] = list(acts.shape)
            for j, layer in enumerate(layers):
                for name in capture.positions:
                    vec = np.asarray(acts[j, positions[name].token_index], dtype=np.float32)
                    if not np.all(np.isfinite(vec)):
                        raise RuntimeError(f"non-finite residual at layer {layer}, {name}")
                    residuals[(layer, name)] = vec.astype(np.dtype(cfg.residual_dtype))

        # B: forced readout
        forced_probs: dict[int, float] | None = None
        if gen.forced_prefix:
            forced = [t for segment in gen.forced_prefix
                      for t in tok.encode(segment, add_special_tokens=False)]
            resp = self._complete(ids + forced, max_tokens=1, logprobs=cfg.top_logprobs)
            top = resp["choices"][0]["logprobs"]["top_logprobs"][0]
            forced_probs, info = probs_from_logprobs(top, values)
            meta["forced_logprobs"] = {**info, "k": cfg.top_logprobs}
        else:
            meta["forced_reason"] = "GenSpec.forced_prefix is empty"

        # C: greedy generation
        if not gen.generate:
            meta["post_reason"] = meta["final_text_reason"] = \
                "generation skipped (GenSpec.generate=False)"
            return RawOutput("", None, None, forced_probs, None, residuals, positions, meta)
        resp = self._complete(ids, max_tokens=gen.max_new_tokens, logprobs=cfg.top_logprobs)
        choice = resp["choices"][0]
        text = choice["text"]
        final_text, reasoning, split_info = split_output(text, cfg.output_format)
        meta.update(split_info)
        meta["finish_reason"] = choice.get("finish_reason")
        meta["hit_max_new_tokens"] = choice.get("finish_reason") == "length"
        post_probs: dict[int, float] | None = None
        lo, hi = min(gen.score_values), max(gen.score_values)
        integer, parse_info = parse_integer(final_text, lo, hi)
        if integer is None:
            meta["post_reason"] = "no parsed score: " + str(parse_info["reason"])
        else:
            char = split_info["final_offset"] + parse_info["match_span"][0]
            lp = choice["logprobs"]
            step = next((i for i, off in enumerate(lp["text_offset"]) if off == char), None)
            if step is None or lp["tokens"][step] != str(parse_info["raw"]):
                meta["post_reason"] = "the parsed digit is not a single clean generation step"
            else:
                post_probs, info = probs_from_logprobs(lp["top_logprobs"][step], values)
                meta["post_logprobs"] = {**info, "k": cfg.top_logprobs}
        return RawOutput(text, final_text, reasoning, forced_probs, post_probs, residuals,
                         positions, meta)
