"""``wbm`` command line.

    wbm run                  score a JSONL of prompts, write readouts + residuals
    wbm validate-september   recompute September's cache and report differences (GPU)
    wbm info                 print package versions

Prompt JSONL, one object per line, either form::

    {"id": "a1", "messages": [{"role": "user", "content": "..."}],
     "spans": {"action": [120, 164]}, "span_message_index": 0}

    {"id": "a2", "template": "... {action} ...", "fields": {"action": "ls"},
     "action_field": "action", "system": null}
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .core.pipeline import compute_readout
from .core.probe import load_probe
from .core.prompt import build_prompt
from .core.types import CaptureSpec, GenSpec, MonitorPrompt
from .core.versions import package_versions


def prompt_from_json(obj: dict) -> MonitorPrompt:
    if "messages" in obj:
        if "spans" not in obj:
            raise ValueError('a prompt with "messages" needs "spans"')
        spans = {name: (int(a), int(b)) for name, (a, b) in obj["spans"].items()}
        return MonitorPrompt(messages=list(obj["messages"]), spans=spans,
                             span_message_index=int(obj.get("span_message_index", 0)))
    if "template" in obj:
        return build_prompt(obj["template"], obj["fields"],
                            action_field=obj.get("action_field", "action"),
                            extra_span_fields=obj.get("extra_span_fields"),
                            system=obj.get("system"))
    raise ValueError('prompt needs "messages" + "spans", or "template" + "fields"')


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def _build_backend(args: argparse.Namespace) -> tuple[Any, GenSpec]:
    from .backends.hf_hooks import HFConfig, HFHooksBackend

    if args.preset == "gpt-oss-20b":
        if not args.template_date:
            raise SystemExit("--preset gpt-oss-20b needs --template-date YYYY-MM-DD")
        backend = HFHooksBackend.gpt_oss_20b(
            template_date=args.template_date, reasoning_effort=args.reasoning_effort,
            attn_chunk_size=args.attn_chunk_size, device=args.device or "cuda:0",
            require_native_quantization=not args.allow_dequantized)
        gen = HFHooksBackend.gpt_oss_genspec(args.max_new_tokens)
    else:
        if not args.model:
            raise SystemExit("pass --model <hub id or local path>, or --preset gpt-oss-20b")
        backend = HFHooksBackend(HFConfig(
            model_id=args.model, dtype=args.dtype, device=args.device or "cpu",
            output_format=args.output_format, template_date=args.template_date))
        gen = GenSpec(max_new_tokens=args.max_new_tokens,
                      forced_prefix=tuple(args.forced_prefix or ["<score>\n"]))
    gen = GenSpec(
        max_new_tokens=gen.max_new_tokens, forced_prefix=gen.forced_prefix,
        score_values=tuple(range(args.score_min, args.score_max + 1)),
        generate=not args.no_generate,
        multi_token_scores="drop" if args.drop_multi_token_scores else "raise")
    return backend, gen


def cmd_run(args: argparse.Namespace) -> int:
    from .store.sidecar import SidecarWriter

    prompts_path = Path(args.prompts)
    if not prompts_path.is_file():
        raise SystemExit(f"prompt file not found: {prompts_path}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    readouts_path = out / "readouts.jsonl"
    if readouts_path.exists() and not args.overwrite:
        raise SystemExit(f"{readouts_path} exists; pass --overwrite or another --out")

    items = [json.loads(line) for line in prompts_path.read_text().splitlines() if line.strip()]
    if args.limit:
        items = items[: args.limit]
    ids = [str(item.get("id", i)) for i, item in enumerate(items)]
    if len(set(ids)) != len(ids):
        raise SystemExit("prompt ids are not unique")

    backend, gen = _build_backend(args)
    layers = _ints(args.layers) if args.layers != "all" else tuple(range(backend.n_layers + 1))
    capture = CaptureSpec(layers=layers, positions=tuple(args.positions.split(",")))
    probe = load_probe(args.probe) if args.probe else None
    run_id = args.run_id or f"wbm-{uuid.uuid4().hex[:12]}"
    run_meta = {
        "run_id": run_id, "started": dt.datetime.now(dt.UTC).isoformat(),
        "prompts": str(prompts_path), "n_prompts": len(items),
        "capture": {"layers": list(capture.layers), "positions": list(capture.positions)},
        "gen": {"max_new_tokens": gen.max_new_tokens, "forced_prefix": list(gen.forced_prefix),
                "score_values": list(gen.score_values), "generate": gen.generate,
                "multi_token_scores": gen.multi_token_scores},
        "backend": backend.base_meta(), "versions": package_versions(),
        "probe": probe.to_meta() if probe else None, "argv": sys.argv[1:],
    }
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=1))

    n_failed = 0
    t0 = time.time()
    with SidecarWriter(out / "acts", dtype=args.store_dtype) as store, \
            readouts_path.open("w", encoding="utf-8") as f:
        for item_id, item in zip(ids, items, strict=True):
            row: dict[str, Any] = {"id": item_id, "run_id": run_id}
            try:
                readout = compute_readout(backend, prompt_from_json(item), capture, gen, probe)
            except Exception as e:  # recorded, counted, and reflected in the exit code
                n_failed += 1
                row.update({"status": "failed", "error": f"{type(e).__name__}: {e}"})
                print(f"{item_id}: FAILED {row['error']}", file=sys.stderr)
            else:
                ref = store.add(run_id, item_id, "prompt", readout.residuals) \
                    if readout.residuals else None
                meta = {k: v for k, v in readout.meta.items() if k != "generated_ids"}
                row.update({
                    "status": "ok", "integer": readout.integer, "expected": readout.expected,
                    "expected_post": readout.expected_post, "probe_score": readout.probe_score,
                    "score_token_probs": readout.score_token_probs,
                    "action_token_index": readout.meta.get("action_token_index"),
                    "activation_ref": ref, "reasoning": readout.reasoning,
                    "completion": readout.completion, "meta": meta,
                })
                e = "None" if readout.expected is None else f"{readout.expected:.3f}"
                print(f"{item_id}: integer {readout.integer} E {e} "
                      f"action token {row['action_token_index']} "
                      f"({readout.meta['n_prompt_tokens']} prompt tokens)")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    run_meta.update({"finished": dt.datetime.now(dt.UTC).isoformat(),
                     "seconds": time.time() - t0, "n_failed": n_failed})
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=1))
    print(f"{len(items) - n_failed} ok, {n_failed} failed -> {out}")
    return 1 if n_failed else 0


def cmd_validate_september(args: argparse.Namespace) -> int:
    from .backends.hf_hooks import HFHooksBackend
    from .september_validate import CacheError, validate_september

    goals = args.goals
    if goals is None:
        if not args.september_repo:
            raise SystemExit("pass --september-repo <checkout of linuxarena-monitor-readout> "
                             "or --goals <side_task_goals.json>")
        goals = Path(args.september_repo) / "experiments" / "logit-curve-replication" / "gpu" \
            / "side_task_goals.json"

    def factory(settings: dict) -> Any:
        if settings["attn_impl"] != "chunked_eager":
            raise SystemExit(f"cache was produced with attn {settings['attn_impl']!r}; only "
                             "chunked_eager is reproduced")
        return HFHooksBackend.gpt_oss_20b(
            template_date=settings["template_date"],
            reasoning_effort=settings["reasoning_effort"],
            attn_chunk_size=settings["attn_chunk_size"], device=args.device)

    try:
        report = validate_september(
            args.cache, goals, factory,
            lambda settings: HFHooksBackend.gpt_oss_genspec(
                args.max_new_tokens or settings["max_new_tokens"]),
            n=args.n, seed=args.seed, stage=args.stage,
            max_prompt_tokens=args.max_prompt_tokens, e_tol=args.e_tol,
            resid_rel_tol=args.resid_rel_tol, progress=lambda s: print(s, flush=True))
    except CacheError as e:
        raise SystemExit(str(e)) from e
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out or f"validate_september_{stamp}.json")
    out.write_text(json.dumps(report, indent=1))
    summary = {k: report[k] for k in ("passed", "checks", "tolerances", "n_run", "counts",
                                      "e_abs_diff", "e_post_abs_diff", "residual_abs_diff",
                                      "residual_max_abs_over_rms")}
    print(json.dumps(summary, indent=1))
    print(f"report -> {out}")
    if not report["passed"]:
        print("VALIDATION FAILED: differences above. Definitions were not adjusted.",
              file=sys.stderr)
    return 0 if report["passed"] else 1


def cmd_info(_args: argparse.Namespace) -> int:
    print(json.dumps(package_versions(), indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wbm", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="score a JSONL of prompts")
    run.add_argument("--prompts", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--model", help="Hugging Face model id or local path")
    run.add_argument("--preset", choices=["gpt-oss-20b"], help="September's settings (GPU)")
    run.add_argument("--layers", default="all", help='comma-separated, 0 = embedding; or "all"')
    run.add_argument("--positions", default="action_last")
    run.add_argument("--max-new-tokens", type=int, default=1024)
    run.add_argument("--forced-prefix", action="append",
                     help="segment force-fed before reading E; repeatable (default '<score>\\n')")
    run.add_argument("--output-format", default="none", choices=["none", "think_tags", "harmony"])
    run.add_argument("--score-min", type=int, default=0)
    run.add_argument("--score-max", type=int, default=10)
    run.add_argument("--drop-multi-token-scores", action="store_true",
                     help="leave out score values that are not single tokens (changes E)")
    run.add_argument("--no-generate", action="store_true", help="E and residuals only")
    run.add_argument("--dtype", default="float32")
    run.add_argument("--device", default=None)
    run.add_argument("--template-date", default=None)
    run.add_argument("--reasoning-effort", default="low")
    run.add_argument("--attn-chunk-size", type=int, default=256)
    run.add_argument("--allow-dequantized", action="store_true")
    run.add_argument("--probe", default=None)
    run.add_argument("--store-dtype", default="float16")
    run.add_argument("--run-id", default=None)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(func=cmd_run)

    val = sub.add_parser("validate-september",
                         help="recompute September's cached readouts (needs GPU + cache)")
    val.add_argument("--cache", required=True,
                     help="September's checkpoints/local-monitor-activations directory")
    val.add_argument("--n", type=int, default=300)
    val.add_argument("--seed", type=int, default=0)
    val.add_argument("--stage", default="main")
    val.add_argument("--september-repo", default=None,
                     help="checkout of linuxarena-monitor-readout (for side_task_goals.json)")
    val.add_argument("--goals", default=None, help="path to side_task_goals.json")
    val.add_argument("--device", default="cuda:0")
    val.add_argument("--max-prompt-tokens", type=int, default=None)
    val.add_argument("--max-new-tokens", type=int, default=None,
                     help="default: the budget recorded in the cache")
    val.add_argument("--e-tol", type=float, default=0.01)
    val.add_argument("--resid-rel-tol", type=float, default=0.02)
    val.add_argument("--out", default=None)
    val.set_defaults(func=cmd_validate_september)

    info = sub.add_parser("info", help="print package versions")
    info.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
