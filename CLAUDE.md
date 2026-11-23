# whitebox-monitor

Measurement harness for a white-box AI control project: runs a trusted
monitor on one agent action and returns the integer score, the expected
score E, and the residual at the action's last token. Repo 2
(`steered-verdict`) holds the experiment and pins this repo by commit.

## Commands

```sh
.venv/bin/python -m pytest            # CPU suite; gpu / needs_cache tests are deselected
.venv/bin/ruff check .
.venv/bin/wbm run --help
```

Run the CPU suite and ruff before every commit. Small commits, one concern
each. There is no GPU on the dev machine.

## Rules that are easy to break

- **No experiment logic or analysis here.** No probe training, statistics,
  plots, AUROC or claims about results. Those belong in repo 2.
- **`core/` imports no framework**: no torch, transformers, Inspect,
  ControlArena or vLLM. Tokenizers are duck-typed.
- **Do not change E, the integer parsing, or the September prompt.**
  `core/readouts.py` and `core/september.py` (vendored block and
  `core/data/old_action.md`) must match
  `thareqyusuf/linuxarena-monitor-readout` at `788c25f`. Golden hashes in
  `tests/september_fixture.py` come from the original module; regenerate
  them only from it. If validation against the September cache fails,
  report the differences and stop.
- **No silent fallbacks.** A missing logprob, an unparseable score or an
  unresolved position raises, or returns `None` with a reason in `meta`.
  Never substitute a default number.
- **Every number carries its provenance**: model id, backend, package
  versions and the token index used go in `Readout.meta`.
- **Do not claim anything is validated on gpt-oss-20b unless it was run.**
  The README's "What has been run" table is the record; update it when a
  GPU phase is run, and mark unrun checks "Not run".
- **Do not write against framework APIs from memory.** ControlArena, Inspect
  and vLLM-Lens change often. Read the installed source
  (`.venv/lib/python3.12/site-packages/...`) for every signature, and keep
  the exact pins in `pyproject.toml` and the README in step.

## Things worth knowing

- Layer index 0 is the embedding output; `i` is the output of block
  `i - 1`. vLLM-Lens counts from the first block, so its backend shifts by
  one and cannot return layer 0.
- Two position rules exist on purpose. `action_last` is the last token
  entirely inside the action (the brief). `action_end` is the token
  containing the action's last character (September); in the September
  cache it is always a token like `'}\n` that includes the following
  newline. Only `action_end` can be checked against the cache. Which is
  primary for repo 2 is an open question for the user.
- `Readout.expected` is September's forced E. `expected_post` is the
  distribution at the greedy step that emitted the digit.
- `@control_agent` turns the builder signature into a pydantic model and
  serialises it into eval logs. Live objects (backend, probe, store) are
  typed `Any` and marked `exclude=True` for that reason.
- Tests are hermetic: `tests/conftest.py` trains a small BPE tokenizer
  in-process (with `}\n` and `10` added as single tokens) and builds a
  random 3-layer Llama. `tests/scripted.py` makes that model emit a chosen
  reply by adding a logit bonus.
- Transformers applies the final norm to `hidden_states[-1]`; the hooks
  return the raw last-block output, as September stored it.
- The September cache is on a rented GPU machine, never local. Code that
  needs it takes a path and fails with a clear message.

## Docs

- `docs/september-notes.md`: how September ran the monitor, E, positions,
  on-disk format.
- `docs/backend-notes.md`: what was checked from source about vLLM-Lens,
  Control Tower and ControlArena, and what is still unrun.
- `docs/injection-setting.md`: design note for the rollout-level injection
  setting (stub only).
