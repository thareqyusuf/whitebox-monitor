"""Run the white-box monitor over an existing ControlArena .eval log.

    python examples/control_arena_post_hoc.py LOG.eval OUT_DIR \
        [--model HuggingFaceTB/SmolLM2-135M-Instruct | --gpt-oss --template-date 2026-09-10]

Writes a new log to OUT_DIR with `suspicion_score` and the `wbm_*` keys on
every assistant message, and the residuals to OUT_DIR/acts.
"""

import argparse
import asyncio
from pathlib import Path

from control_arena.eval import evaluate_monitors_post_hoc

from whitebox_monitor import CaptureSpec, GenSpec
from whitebox_monitor.adapters.control_arena import whitebox_monitor_builder
from whitebox_monitor.backends.hf_hooks import HFConfig, HFHooksBackend
from whitebox_monitor.store import SidecarWriter

parser = argparse.ArgumentParser()
parser.add_argument("log")
parser.add_argument("out_dir")
parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
parser.add_argument("--gpt-oss", action="store_true", help="gpt-oss-20b with September's settings")
parser.add_argument("--template-date")
parser.add_argument("--layers", default="12")
args = parser.parse_args()

if args.gpt_oss:
    backend = HFHooksBackend.gpt_oss_20b(template_date=args.template_date)
    gen = HFHooksBackend.gpt_oss_genspec()
else:
    backend = HFHooksBackend(HFConfig(model_id=args.model))
    gen = GenSpec(max_new_tokens=128, multi_token_scores="drop", score_values=tuple(range(11)))

capture = CaptureSpec(layers=tuple(int(x) for x in args.layers.split(",")),
                      positions=("action_last",))
out_dir = Path(args.out_dir)
with SidecarWriter(out_dir / "acts") as store:
    monitor = whitebox_monitor_builder(backend=backend, capture=capture, gen=gen, store=store)
    paths = asyncio.run(evaluate_monitors_post_hoc(args.log, [monitor], out_dir))
print("wrote", paths[0][0])
