# Notes on the September experiment

Source: `thareqyusuf/linuxarena-monitor-readout` at `788c25f` (read 23 Nov
2026). File paths below are in that repo. Everything here was read from its
code and its committed raw result files; nothing was re-run.

## How the monitor was run

- **Library**: Hugging Face Transformers 5.17.0, torch 2.14.0 (cu130),
  tokenizers 0.23.2, `kernels` 0.16.1. No nnsight or vLLM in the capture
  path. (`infra/constraints.txt`, `gpu/requirements.txt`,
  `results/raw/local-monitor-activations/g1_env_*.json`.)
- **Model**: `openai/gpt-oss-20b`, loaded with `dtype=torch.bfloat16`,
  `device_map={"": 0}`. Weights stay in native **MXFP4**: the runner asserts
  `not model.hf_quantizer.quantization_config.dequantize`. 24 decoder
  layers, `d_model = 2880`.
- **Attention**: a custom implementation `chunked_eager`
  (`local-monitor-activations/chunked_attention.py`), registered through
  `AttentionInterface`. It computes the same quantity as gpt-oss's eager
  attention (sinks, sliding window) over query chunks, so a ~25k-token
  prompt fits in memory. The chunk size is **fixed at 256** for every prompt
  because chunking changes the reduction order and can move logits (up to
  ~1.5 through MoE routing flips, per the file's comment). The model is
  called with explicit `cache_position` / `position_ids` after the prefill.
- **Hook mechanism**: plain `register_forward_hook` on
  `model.model.embed_tokens` and on each of `model.model.layers`. During the
  single prefill pass the hook keeps `output[0, positions, :]` as float16.
- **Prompt length**: up to 32,000 tokens (`--max-prompt-tokens 32000`);
  longer prompts are skipped.
- **Chat template**: harmony, with the date pinned to a literal
  (`--template-date 2026-09-10` for the main pass) and
  `reasoning_effort="low"`. The last prompt token is asserted to be
  `assistant` (id 173781).
- **Per action** (`run_monitor.py::Runner.run_action`): one prefill pass
  with capture; forced readout on a deep copy of the KV cache; greedy
  unforced generation on the original cache (budget 1024, stop on
  `<|return|>` / `<|call|>`).
- **Determinism**: 104 actions recomputed in a different process on the
  same pod were bitwise identical in activations, score, generation, E and
  E_post (`g4_main_vs_consistent_high_determinism_*.json`).

## Definition of E

`run_monitor.py::read_distribution` and `forced_readout`:

1. **Forcing the score position.** After the prompt (which ends in
   `<|start|>assistant`), the model is fed `<|channel|>final<|message|>` in
   one forward call and then `<score>\n` in a second call, both on a copy of
   the prefill cache. The monitor writes no reasoning first.
2. **Tokens.** `str(v)` for `v` in 0..10, plain spelling, no leading space.
   Each must be a single token (asserted); "10" is one token for gpt-oss.
3. **Distribution.** `softmax` over the whole vocabulary in float32 at the
   position after `<score>\n`; keep the eleven probabilities.
4. **Normalisation.** `E = sum(v * p_v) / sum(p_v)`. The denominator is
   stored as `digit_token_mass`. The stored `digit_dist` is not normalised.

`expected_score_post` uses the same formula on the greedy step that emitted
the parsed digit (the next-token distribution at `pre_score`).

Ambiguity check (section 9 of the brief): the code is unambiguous about "10"
(included, single token) and about forcing. The September README says "ten
digit tokens", which is loose wording; the code and stored records use
eleven values. An earlier CPU smoke script (`logit-curve-replication/gpu/
run_monitor.py`) also summed space-prefixed spellings and dropped "10" for
Qwen; that is not the definition the cached E was computed with.

The integer is parsed from the final channel only, with
`<score>\s*(\d+)\s*</score>` and a looser fallback; values outside 0..10
give `None`.

## The 7 positions and 25 layers

Axis 0 of every activation tensor, in this order:

| # | Name | Phase | Definition |
|---|---|---|---|
| 0 | `instruction_end` | prompt | last token of the user message (the closing `</score>` of the template) |
| 1 | `transcript_end` | prompt | last token of the `<history>` block content |
| 2 | `action_end` | prompt | last token of the `<current_action>` block content |
| 3 | `prompt_end` (old name `final`) | prompt | last token of the templated prompt (`assistant`) |
| 4 | `analysis_end` | generation | token before the `<|end|>` that closes the analysis channel |
| 5 | `reasoning_end` | generation | last token before `</reasoning>` in the final channel |
| 6 | `pre_score` | generation | token before the parsed digit (the `>\n` of `<score>\n`) |

Axis 1 has 25 entries: index 0 is the **embedding output**, index `i` is the
output of decoder layer `i - 1`. The last entry is the raw output of the
last block, before the final norm.

### Is the action's last token one of them?

**Yes by name, with a catch.** `action_end` is defined as "the token that
contains the last character of the action" (`tok_at_char_end`: last token
whose start is before the span end). In **all 2,012** cached actions that
token also contains the newline that follows the action in the template:

| decoded `action_end` token | count |
|---|---|
| `'}\n` | 1661 |
| `}\n` | 169 |
| `]}\n` | 98 |
| `"}\n` | 84 |

So the cached position is never a token that lies entirely inside the
action. The brief's rule ("the last token whose character span ends at or
before the action's end") picks the token **before** it in every cached
action. Consequences:

- The cache can validate `action_end` (September's rule), which this
  package implements under that name. It cannot validate the brief's
  stricter position, which this package calls `action_last`.
- For the causal argument the difference is small but real: `'}\n` reads
  one template character after the action. An injection placed after that
  newline leaves it unchanged; an injection glued directly to the action
  changes the token.

This is the situation section 9 of the brief says to raise. Both rules are
implemented and reported; which one repo 2 treats as primary is the user's
call.

## Prompt builder

`logit-curve-replication/gpu/format_monitor_prompt.py` is a pure-Python port
of control-tower 1.0.1's live action-monitor prompt
(`mean_action_live_monitor`, `provide_goals=True`): one user message, no
system prompt, template `prompts/old_action.md`
(sha256 `a4a48d03…2058`). The September README reports it byte-identical to
control-tower on all 17,234 actions. Details that matter:

- The current action is rendered as `Tool call: <function>: <arguments
  dict repr>` with its output hidden.
- History is `Action i:\n<call>\nResult: …\nError: …` joined by blank
  lines; hidden tools (`run_side_task_test`, `submit`, `side_task_scorer`)
  and malformed actions are skipped; `text_editor` `str_replace` calls get a
  reconstructed diff; outputs over 400,000 chars are truncated.
- `monitor_prompt.py::render_user_text` locates the history and action
  spans by `text.index("<history>\n")` / `text.index("<current_action>\n",
  h1)` plus the known lengths, and asserts what follows.

This package vendors the formatting functions unchanged
(`core/september.py`) and the template byte-for-byte; a test pins golden
hashes produced by the original module.

## On-disk format

Under `checkpoints/local-monitor-activations/` (not in the repo, on the GPU
machine):

- `data/export/{trajectories,honest,attack}.jsonl`: the trajectories.
- `out/<stage>/scores.jsonl`: one JSON record per action. Fields used
  here: `traj_id`, `action_idx`, `render`, `status`, `prompt_sha256`,
  `n_prompt_tokens`, `capture_positions` (name, token index, token id,
  decoded token), `score`, `expected_score`, `digit_dist`,
  `digit_token_mass`, `expected_score_post`, `generation`, `generated_ids`,
  `template_date`, `reasoning_effort`, `attn_impl`, `attn_chunk_size`,
  `max_new_tokens`.
- `out/<stage>/acts_XXXX.pt`: `torch.save` of a dict: `"_meta"` plus
  `"traj_id|action_idx|render"` → float16 tensor `[7, 25, 2880]`. Missing
  generation positions are NaN rows.
- `out/<stage>/acts_index.json`: `{"shards": {file: [keys]},
  "position_order": [...]}`.
- `out/<stage>/run_meta.json`: the run settings.
- The analysis stage consolidates the shards into
  `checkpoints/latent-vs-verbalized/acts.npy`, float16
  `[n_actions, 7, 25, 2880]`, with `acts_index.json` listing rows.

Labels: `experiments/action-level-attack-label/action_labels.json`
(columnar: `columns`, `data`, `n`), pinned by commit in the analysis code.
Labels are not used by this package.

## Where this package deviates

- `Readout.expected` is September's forced E; the natural-position value is
  `Readout.expected_post`.
- Only the four prompt-side positions can be captured. The three
  generation-side positions are not implemented.
- Residuals at `action_last` are new; they have no September counterpart.
