# Backend notes

Status of each item the brief listed as unchecked. "Source" means read from
code on 23 Nov 2026, not run. Nothing in this file was run on a GPU.

Sources: vLLM-Lens `v1.3.0` (commit `3d11fb0`), control-tower `v6.2.0`
(commit `06111c1`), control-arena 20.1.1 and inspect-ai 0.3.291 as installed.

## vLLM-Lens

| Question | Answer | Basis |
|---|---|---|
| Works with gpt-oss-20b (harmony, MXFP4)? | **Not run. Not settled by source.** gpt-oss-20b appears only in an opt-in attention Q/K parity test, not for residual capture or the chat path. No mention of harmony or MXFP4. Its compatibility doc says passing the small Qwen suite does not establish support for other models or quantization modes. | `vllm_lens/tests/test_qk_parity_models.py`, `docs/compatibility.md` |
| MoE? | Layer discovery ignores `FusedMoE` entries and keys on attention modules; the README reports an MoE model working for another lens. Not run for gpt-oss. | `_worker_ext.py:64-73`, README |
| Do logprobs come back alongside activations? | Raw `/v1/completions`: yes, the plugin adds keys and leaves the rest of the response alone. Inspect provider: expected through inheritance from Inspect's `VLLMAPI`, but no test in the repo covers it. **Not run.** | `client.py`, `inspect_provider.py:130` |
| Can capture be limited to token positions? | **No.** Layer selection only. The full `(n_layers, n_positions, d)` tensor is returned in the model's dtype. A custom hook (`apply_hooks`) could save one row, but it sees chunks under chunked prefill with no absolute offset, and hook errors are swallowed. This package captures few layers and slices client-side. | `_worker_ext.py:582-585` |
| Are prompt-token activations returned? | **Yes.** Positions `0 .. n_prompt + n_gen - 2`; index 0 is the first prompt token as the server tokenized it, including template tokens. Prefix caching is turned off for capture requests. | `_activations_plugin.py:442-481` |
| Layer convention | vLLM-Lens layer `i` is the output of block `i` (HF `hidden_states[i + 1]`). This package's layer 0 is the embedding output, so layer `L` is requested as `L - 1` and layer 0 is unavailable. | `tests/test_activations_match.py:55` |
| Request shape | `vllm_xargs: {"output_residual_stream": "[15]"}` (JSON string over HTTP); response `activations.residual_stream` = base64 + zstd, with dtype and shape. | `_serialize.py`, `tests/test_server.py` |
| Versions | `vllm==0.30.0` exactly; validated with torch 2.13.0 and transformers 5.18.0, which differ from September's 2.14.0 / 5.17.0. Use a separate environment. | `pyproject.toml`, `docs/compatibility.md` |

Known differences the parity test has to measure, not assume away:

- E from the server is built from the top-k logprobs (k = 20 here), not the
  full softmax. Missing score tokens count as zero and are listed in `meta`.
- The forced prefix goes in one request with the prompt instead of separate
  cached forward calls. September noted that chunking order can move logits.
- Whether vLLM's forward pass on MXFP4 weights matches Transformers'
  `chunked_eager` numerically is unknown.

`backends/vllm_lens.py` is tested only against a fake transport that
behaves as the source says the server does.

## Control Tower (LinuxArena's framework)

Out of scope; recorded for later. At `v6.2.0` monitors live in
`src/control_tower/monitors/`. The path September ported from
(`protocols/monitoring/action_monitors/`, control-tower 1.0.1) no longer
exists.

- A monitor is `async def monitor(model_calls, *, ...) -> inspect_ai.scorer.Score`,
  registered with `@monitors.register()`, or a prompt `.md` with
  `{transcript}` and `{action}` placeholders.
- The model answers through a `score` tool with a 0..100 score, not a
  `<score>` tag with 0..10.
- Scores are stored in the sample store (`MonitorScores.suspicion_scores`),
  not in message metadata.

So the September prompt and score scale are not what current Control Tower
uses by default. An adapter would need its own prompt handling.

## ControlArena: differences from the brief's section 4

Checked against the installed source.

- `MonitorResponse` has four fields, not two: `suspicion_score`, `reason`,
  `model_usage`, `score_parsed`. `score_parsed=False` is how ControlArena
  marks an unusable score; the adapter uses it.
- The `@monitor` decorator and the post-hoc runner write to the **last
  assistant message**, found by searching backwards, not to `messages[-1]`.
- On an unparseable score the built-in monitor returns
  `default_suspicion_score` (0.0 by default) with `score_parsed=False`. The
  adapter does the same and also sets `wbm_integer=None`,
  `wbm_integer_parsed=False`.
- Post-hoc runs set `sample_id="post_hoc"` for every sample, so sidecar keys
  rely on the message id to tell actions apart.
- Agent configs are serialised into the eval log. The backend, probe and
  store are excluded from that dump.
- Replaying a log needs the `tool_output_limit` recorded in its solver
  plan; ControlArena raises if it cannot recover it.

None of these changes the design. They are listed because the brief asks to
flag any difference.

## HF hooks backend

- Run on CPU with a random 3-layer Llama and with
  `HuggingFaceTB/SmolLM2-135M-Instruct`. Residuals are checked against
  `output_hidden_states`, and forced E against a single uncached pass.
- The `gpt_oss_20b()` preset (MXFP4, `chunked_eager`, harmony) has **not
  been run**. Open risk: it assumes gpt-oss's `forward` accepts
  `cache_position`, `position_ids` and `logits_to_keep` as in transformers
  5.17.0, which is what September ran.
- The last layer index is the raw block output. Transformers'
  `hidden_states[-1]` has the final norm applied, so the two differ there.
