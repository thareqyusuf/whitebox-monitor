# Rollout-level injection setting (design note, not built)

`core/inject.py` edits a finished monitor prompt. That is enough to test a
readout, but it is not how an attacker reaches a monitor: in a real rollout
the text arrives through the environment. This note sketches the later
phase. `adapters/injection_setting.py` is a stub that raises.

## What it should do

Wrap an existing ControlArena `Setting` so that chosen text is planted where
the agent's actions will carry it into the monitor's view:

| Channel | Mechanism | Reaches the monitor through |
|---|---|---|
| File in the sandbox (README, config, comment) | write the file in the setting's setup solver | tool output when the agent reads it (history) |
| Tool output | wrap a tool so its result has the text appended | history |
| Agent-authored action | attack policy instructed to include the text in its tool call | the action itself |

The first two put the text **before** the action under review in later
steps (history), the third puts it **inside** the action. `after_action`
has no rollout analogue with ControlArena's default prompt; it exists in
`inject` as the causal control.

## Sketch

- `injection_setting_builder(base: Setting, plan: InjectionPlan) -> Setting`,
  where `InjectionPlan` lists `(channel, path_or_tool, text, when)`.
- Record every planted string and where it went in sample metadata
  (`wbm_injection`), so analysis can find the monitor steps that saw it
  without string search.
- Templates live in repo 2 (`steered-verdict`); this repo only places them.
- The prompt builder must still return the `action` span. Injected text in
  history does not move it, since spans come from offset arithmetic.

## Open questions

- Whether to plant text at setup time or at a chosen step (needs a tool
  wrapper with step state).
- Honest runs need the same planted text to give a matched control.
- ControlArena's monitor view hides attack tools
  (`convert_agent_state_to_monitor_view`); a tool-output wrapper must sit
  below that conversion so the monitor sees what the agent saw.
