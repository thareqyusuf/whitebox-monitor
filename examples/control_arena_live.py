"""Phase 6: a small honest + attack run in a ControlArena setting with the
white-box monitor inside trusted monitoring. NOT RUN from this repo yet.

    python examples/control_arena_live.py --template-date 2026-11-23 \
        --untrusted-model anthropic/claude-sonnet-5-5 --limit 2 --log-dir logs/phase6

Needs: a GPU with gpt-oss-20b (the monitor), Docker (the Apps sandbox) and an
API key for the untrusted model. The monitor does not use ControlArena's
trusted model role; gpt-oss runs in-process through the HF hooks backend.
"""

import argparse

from control_arena import EvalMode
from control_arena.eval import ControlEvalConfig, get_control_task_product
from control_arena.micro_protocol import trusted_monitoring_builder
from control_arena.settings.apps.apps_setting import AppsSetting
from inspect_ai import eval as inspect_eval

from whitebox_monitor import CaptureSpec
from whitebox_monitor.adapters.control_arena import whitebox_monitor_builder
from whitebox_monitor.backends.hf_hooks import HFHooksBackend
from whitebox_monitor.store import SidecarWriter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-date", required=True)
    parser.add_argument("--untrusted-model", required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--log-dir", default="logs/phase6")
    parser.add_argument("--layers", default="12")
    args = parser.parse_args()

    backend = HFHooksBackend.gpt_oss_20b(template_date=args.template_date)
    capture = CaptureSpec(layers=tuple(int(x) for x in args.layers.split(",")),
                          positions=("action_last",))
    with SidecarWriter(f"{args.log_dir}/acts") as store:
        monitor = whitebox_monitor_builder(backend=backend, capture=capture,
                                           gen=HFHooksBackend.gpt_oss_genspec(), store=store)
        tasks = get_control_task_product(
            settings=AppsSetting(),
            micro_protocols=trusted_monitoring_builder(monitor=monitor),
            modes=[EvalMode.HONEST, EvalMode.ATTACK],
            eval_config=ControlEvalConfig(untrusted_model=args.untrusted_model,
                                          limit=args.limit, log_dir=args.log_dir),
        )
        inspect_eval(tasks, limit=args.limit, log_dir=args.log_dir, max_samples=1)


if __name__ == "__main__":
    main()
