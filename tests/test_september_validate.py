"""Plumbing test for `wbm validate-september` on a SYNTHETIC cache.

The cache is written in September's on-disk layout by the tiny CPU model, so
this checks the reading, sampling, comparison and pass/fail logic. It says
nothing about gpt-oss-20b: the real check is the `gpu` + `needs_cache` test
below, run by the user on a pod.
"""

import json
import os

import numpy as np
import pytest
import torch
from september_fixture import TRAJ

from whitebox_monitor import CaptureSpec, GenSpec, compute_readout
from whitebox_monitor.core import september as sep
from whitebox_monitor.september_validate import PROMPT_POSITIONS, CacheError, validate_september

SETTINGS = {"template_date": None, "reasoning_effort": "low", "attn_impl": "chunked_eager",
            "attn_chunk_size": 256, "max_new_tokens": 6}
GEN = GenSpec(max_new_tokens=6)
ORDER = ["instruction_end", "transcript_end", "action_end", "prompt_end",
         "analysis_end", "reasoning_end", "pre_score"]


@pytest.fixture()
def cache(tmp_path, backend):
    export = tmp_path / "cache" / "data" / "export"
    stage = tmp_path / "cache" / "out" / "main"
    export.mkdir(parents=True)
    stage.mkdir(parents=True)
    goals = tmp_path / "side_task_goals.json"
    goals.write_text(json.dumps({"goals_text": {"demo": TRAJ["side_task_goals"]}}))
    meta = {k: TRAJ[k] for k in ("traj_id", "env", "main_task_description",
                                 "environment_description")}
    (export / "trajectories.jsonl").write_text(json.dumps({**meta, "side": "attack"}) + "\n")
    (export / "honest.jsonl").write_text("")
    with (export / "attack.jsonl").open("w") as f:
        for i, a in enumerate(TRAJ["actions"]):
            f.write(json.dumps({"traj_id": TRAJ["traj_id"], "action_idx": i,
                                "tool_name": a["function"], "tool_input": a["arguments"],
                                "tool_output": a["output"], "output_error": a["output_error"]}) + "\n")
    capture = CaptureSpec(layers=tuple(range(backend.n_layers + 1)), positions=PROMPT_POSITIONS)
    shard, records = {}, []
    for i in range(len(TRAJ["actions"])):
        prompt = sep.september_prompt(TRAJ, i)
        r = compute_readout(backend, prompt, capture, GEN)
        acts = np.full((7, backend.n_layers + 1, 32), np.nan, dtype=np.float16)
        for j, name in enumerate(PROMPT_POSITIONS):
            for layer in range(backend.n_layers + 1):
                acts[j, layer] = r.residuals[(layer, name)].astype(np.float16)
        key = f"{TRAJ['traj_id']}|{i}|original"
        shard[key] = torch.from_numpy(acts)
        records.append({
            "traj_id": TRAJ["traj_id"], "action_idx": i, "render": "original", "status": "ok",
            "prompt_sha256": sep.prompt_sha256(prompt), "n_prompt_tokens": r.meta["n_prompt_tokens"],
            "capture_positions": [{"name": n, "token_index": r.meta["positions"][n]["token_index"]}
                                  for n in PROMPT_POSITIONS],
            "score": None if r.integer is None else float(r.integer),
            "expected_score": r.expected, "expected_score_post": r.expected_post,
            "generation": r.completion, **SETTINGS})
    records.append({"traj_id": TRAJ["traj_id"], "action_idx": 0, "render": "neutralized",
                    "status": "ok", **SETTINGS})
    (stage / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    torch.save({"_meta": {"position_order": ORDER}, **shard}, stage / "acts_0000.pt")
    (stage / "acts_index.json").write_text(json.dumps(
        {"shards": {"acts_0000.pt": list(shard)}, "position_order": ORDER}))
    return tmp_path / "cache", goals, stage


def _run(cache_dir, goals, backend, **kw):
    return validate_september(cache_dir, goals, lambda settings: backend, lambda settings: GEN, **kw)


def test_identical_backend_reproduces_the_cache(cache, backend):
    cache_dir, goals, _ = cache
    report = _run(cache_dir, goals, backend, n=5, seed=1)
    assert report["passed"] and all(report["checks"].values())
    assert report["n_run"] == 5 and report["n_sampled"] == 5
    assert report["e_abs_diff"]["max"] == 0.0
    assert report["counts"]["bitwise_equal_residual_actions"] == 5
    assert report["counts"]["integer_mismatch"] == 0 and report["counts"]["generation_mismatch"] == 0
    assert all(v["max"] == 0.0 for v in report["residual_abs_diff"].values())
    # the brief's action_last is not September's cached action_end here either
    assert report["counts"]["action_last_differs_from_action_end"] == 5
    assert report["run_settings"] == SETTINGS
    json.dumps(report)


def test_all_records_when_n_exceeds_cache(cache, backend):
    cache_dir, goals, _ = cache
    assert _run(cache_dir, goals, backend, n=300)["n_run"] == len(TRAJ["actions"])


def test_differences_are_reported_and_fail(cache, backend):
    cache_dir, goals, stage = cache
    payload = torch.load(stage / "acts_0000.pt", weights_only=True)
    key = f"{TRAJ['traj_id']}|2|original"
    payload[key][2, 1] += 0.5                       # action_end, layer 1
    torch.save(payload, stage / "acts_0000.pt")
    rows = [json.loads(line) for line in (stage / "scores.jsonl").read_text().splitlines()]
    rows[3]["expected_score"] += 0.25
    (stage / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = _run(cache_dir, goals, backend, n=300)
    assert not report["passed"]
    assert not report["checks"]["e_within_tolerance"]
    assert not report["checks"]["residuals_within_tolerance"]
    assert report["e_abs_diff"]["max"] == pytest.approx(0.25)
    assert report["residual_abs_diff"]["action_end"]["max"] == pytest.approx(0.5, abs=1e-2)
    assert report["residual_abs_diff"]["prompt_end"]["max"] == 0.0
    assert report["counts"]["bitwise_equal_residual_actions"] == len(TRAJ["actions"]) - 1


def test_prompt_mismatch_is_not_run(cache, backend):
    cache_dir, goals, stage = cache
    rows = [json.loads(line) for line in (stage / "scores.jsonl").read_text().splitlines()]
    rows[0]["prompt_sha256"] = "0" * 64
    (stage / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = _run(cache_dir, goals, backend, n=300)
    assert not report["passed"] and report["counts"]["prompt_sha_mismatch"] == 1
    assert report["n_run"] == len(TRAJ["actions"]) - 1


def test_missing_cache_message(tmp_path, backend):
    with pytest.raises(CacheError, match="GPU machine"):
        _run(tmp_path / "absent", tmp_path / "goals.json", backend)


@pytest.mark.gpu
@pytest.mark.needs_cache
def test_september_reproduction_on_gpt_oss_20b():
    """Phase 4. Run on the pod:

        WBM_SEPTEMBER_CACHE=... WBM_SEPTEMBER_REPO=... pytest -m "gpu and needs_cache"
    """
    from whitebox_monitor.cli import main

    cache_dir = os.environ["WBM_SEPTEMBER_CACHE"]
    repo = os.environ["WBM_SEPTEMBER_REPO"]
    assert main(["validate-september", "--cache", cache_dir, "--september-repo", repo,
                 "--n", os.environ.get("WBM_VALIDATE_N", "300")]) == 0
