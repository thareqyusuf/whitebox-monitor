"""Phase 4: recompute September's cached numbers and report the differences.

Reads September's on-disk layout (``linuxarena-monitor-readout`` @ 788c25f)::

    <cache>/data/export/{trajectories,honest,attack}.jsonl   trajectories
    <cache>/out/<stage>/scores.jsonl                         one record per action
    <cache>/out/<stage>/acts_index.json                      shard -> keys
    <cache>/out/<stage>/acts_XXXX.pt                         key -> fp16 [7, 25, 2880]

where ``<cache>`` is September's ``checkpoints/local-monitor-activations``.
Activation axis 0 is ``position_order`` from the index, axis 1 is
embedding + 24 blocks.

For a seeded sample of ``status == "ok"``, ``render == "original"`` records
it rebuilds the prompt, runs the backend, and compares

    prompt sha256            exact
    capture token indices    exact, for the four prompt-side positions
    integer                  exact
    generated text           exact (reported, not part of pass/fail)
    E, E_post                absolute difference
    residuals                at the four prompt-side positions, all layers,
                             both sides as float16: absolute difference, and
                             difference relative to the cached vector's RMS

The three generation-side positions are not recomputed. This module only
reports. It never changes a definition to make numbers agree.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .core import september as sep
from .core.pipeline import compute_readout
from .core.types import CaptureSpec, GenSpec
from .core.versions import package_versions

PROMPT_POSITIONS = ("instruction_end", "transcript_end", "action_end", "prompt_end")
#: Stated tolerances. September's own rerun in a different process on the
#: same pod was bitwise identical (g4_main_vs_consistent_high_determinism),
#: so on the same hardware and software the expected difference is zero.
#: These bounds allow for a different GPU model, nothing more.
E_TOL = 0.01          # absolute, on the 0-10 scale
RESID_REL_TOL = 0.02  # max |diff| / RMS(cached vector)


class CacheError(FileNotFoundError):
    pass


def _need(path: Path, what: str) -> Path:
    if not path.exists():
        raise CacheError(
            f"{what} not found: {path}\nThe September cache lives on the GPU machine "
            "(checkpoints/local-monitor-activations in linuxarena-monitor-readout). "
            "Pass its path with --cache."
        )
    return path


def load_records(stage_dir: Path) -> list[dict]:
    records = []
    with _need(stage_dir / "scores.jsonl", "scores.jsonl").open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def sample_records(records: list[dict], indexed: set[str], n: int, seed: int,
                   max_prompt_tokens: int | None = None) -> list[dict]:
    ok = [r for r in records if r.get("status") == "ok" and r.get("render") == "original"
          and f"{r['traj_id']}|{r['action_idx']}|original" in indexed
          and (max_prompt_tokens is None or r["n_prompt_tokens"] <= max_prompt_tokens)]
    ok.sort(key=lambda r: (r["traj_id"], r["action_idx"]))
    if not ok:
        raise ValueError("no usable records (status ok, render original, activations present)")
    if n >= len(ok):
        return ok
    return sorted(random.Random(seed).sample(ok, n), key=lambda r: (r["traj_id"], r["action_idx"]))


class ShardReader:
    def __init__(self, stage_dir: Path) -> None:
        self.dir = stage_dir
        index = json.loads(_need(stage_dir / "acts_index.json", "acts_index.json").read_text())
        self.position_order: list[str] = [
            {"final": "prompt_end"}.get(p, p) for p in index["position_order"]]
        self.shard_of = {k: shard for shard, keys in index["shards"].items() for k in keys}
        self._loaded: tuple[str, dict] | None = None

    def get(self, key: str) -> np.ndarray:
        import torch

        shard = self.shard_of[key]
        if self._loaded is None or self._loaded[0] != shard:
            payload = torch.load(_need(self.dir / shard, "activation shard"),
                                 map_location="cpu", weights_only=True)
            self._loaded = (shard, payload)
        return self._loaded[1][key].numpy()


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "max": None, "mean": None}
    return {"n": len(values), "max": float(max(values)), "mean": float(np.mean(values))}


def validate_september(
    cache: str | Path,
    goals_path: str | Path,
    backend_factory: Callable[[dict], Any],
    gen_factory: Callable[[dict], GenSpec],
    *,
    n: int = 300,
    seed: int = 0,
    stage: str = "main",
    max_prompt_tokens: int | None = None,
    e_tol: float = E_TOL,
    resid_rel_tol: float = RESID_REL_TOL,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the comparison and return the report (also see ``passed``).

    ``backend_factory(run_settings)`` builds the backend from the settings the
    cache was produced with (template date, reasoning effort, chunk size);
    ``gen_factory(run_settings)`` builds the GenSpec (forced prefix, budget).
    """
    cache = Path(cache)
    stage_dir = _need(cache / "out" / stage, "stage directory")
    export_dir = _need(cache / "data" / "export", "export directory")
    _need(Path(goals_path), "side_task_goals.json")
    shards = ShardReader(stage_dir)
    records = sample_records(load_records(stage_dir), set(shards.shard_of), n, seed,
                             max_prompt_tokens)
    settings_keys = ("template_date", "reasoning_effort", "attn_impl", "attn_chunk_size",
                     "max_new_tokens")
    settings = {k: records[0].get(k) for k in settings_keys}
    for r in records:
        if any(r.get(k) != settings[k] for k in settings_keys):
            raise ValueError(f"sampled records were produced with mixed settings: "
                             f"{ {k: r.get(k) for k in settings_keys} } vs {settings}")
    trajs = sep.load_export(export_dir, goals_path)
    backend = backend_factory(settings)
    n_layers = backend.n_layers
    gen = gen_factory(settings)
    capture = CaptureSpec(layers=tuple(range(n_layers + 1)),
                          positions=PROMPT_POSITIONS + ("action_last",))

    rows: list[dict] = []
    e_diffs: list[float] = []
    epost_diffs: list[float] = []
    resid_abs: dict[str, list[float]] = {p: [] for p in PROMPT_POSITIONS}
    resid_mean_abs: dict[str, list[float]] = {p: [] for p in PROMPT_POSITIONS}
    resid_rel: dict[str, list[float]] = {p: [] for p in PROMPT_POSITIONS}
    counts = {"prompt_sha_mismatch": 0, "token_index_mismatch": 0, "integer_mismatch": 0,
              "generation_mismatch": 0, "e_missing": 0, "bitwise_equal_residual_actions": 0,
              "action_last_differs_from_action_end": 0}

    for i, rec in enumerate(records):
        key = f"{rec['traj_id']}|{rec['action_idx']}|original"
        row: dict[str, Any] = {"key": key, "n_prompt_tokens": rec["n_prompt_tokens"]}
        prompt = sep.september_prompt(trajs[rec["traj_id"]], rec["action_idx"])
        sha = sep.prompt_sha256(prompt)
        row["prompt_sha256_match"] = sha == rec["prompt_sha256"]
        if not row["prompt_sha256_match"]:
            counts["prompt_sha_mismatch"] += 1
            row["skipped"] = "prompt sha256 differs from the cache; not run"
            rows.append(row)
            continue
        readout = compute_readout(backend, prompt, capture, gen)
        cached_pos = {{"final": "prompt_end"}.get(c["name"], c["name"]): c
                      for c in rec["capture_positions"]}
        idx_ok = True
        for name in PROMPT_POSITIONS:
            ours = readout.meta["positions"][name]["token_index"]
            if ours != cached_pos[name]["token_index"]:
                idx_ok = False
        row["token_indices_match"] = idx_ok
        counts["token_index_mismatch"] += not idx_ok
        row["action_last_minus_action_end"] = (
            readout.meta["positions"]["action_last"]["token_index"]
            - readout.meta["positions"]["action_end"]["token_index"])
        counts["action_last_differs_from_action_end"] += row["action_last_minus_action_end"] != 0

        cached_int = None if rec.get("score") is None else int(rec["score"])
        row["integer"] = {"ours": readout.integer, "cached": cached_int}
        counts["integer_mismatch"] += readout.integer != cached_int
        row["generation_match"] = readout.completion == rec.get("generation")
        counts["generation_mismatch"] += not row["generation_match"]

        if readout.expected is None or rec.get("expected_score") is None:
            counts["e_missing"] += 1
            row["e_abs_diff"] = None
        else:
            row["e_abs_diff"] = abs(readout.expected - rec["expected_score"])
            e_diffs.append(row["e_abs_diff"])
        if readout.expected_post is not None and rec.get("expected_score_post") is not None:
            row["e_post_abs_diff"] = abs(readout.expected_post - rec["expected_score_post"])
            epost_diffs.append(row["e_post_abs_diff"])

        cached = shards.get(key)
        if cached.shape[1] != n_layers + 1:
            raise ValueError(f"cached activations have {cached.shape[1]} layers, model has "
                             f"{n_layers + 1} (embedding + blocks)")
        bitwise = idx_ok
        row["residual_max_abs_diff"] = {}
        for name in PROMPT_POSITIONS:
            c = cached[shards.position_order.index(name)].astype(np.float32)
            ours = np.stack([readout.residuals[(layer, name)].astype(np.float16)
                             for layer in range(n_layers + 1)]).astype(np.float32)
            diff = np.abs(ours - c)
            rms = np.sqrt(np.mean(c ** 2, axis=1))
            rel = diff.max(axis=1) / np.where(rms > 0, rms, np.inf)
            resid_abs[name].append(float(diff.max()))
            resid_mean_abs[name].append(float(diff.mean()))
            resid_rel[name].append(float(rel.max()))
            row["residual_max_abs_diff"][name] = float(diff.max())
            bitwise = bitwise and bool(np.array_equal(ours, c))
        counts["bitwise_equal_residual_actions"] += bitwise
        rows.append(row)
        if progress:
            progress(f"[{i + 1}/{len(records)}] {key} E diff {row['e_abs_diff']} "
                     f"resid max {max(row['residual_max_abs_diff'].values()):.4g}")

    n_run = sum(1 for r in rows if "skipped" not in r)
    e = _stats(e_diffs)
    rel_max = max((max(v) for v in resid_rel.values() if v), default=None)
    checks = {
        "all_prompts_match": counts["prompt_sha_mismatch"] == 0,
        "all_token_indices_match": counts["token_index_mismatch"] == 0,
        "e_within_tolerance": e["max"] is not None and e["max"] <= e_tol and counts["e_missing"] == 0,
        "residuals_within_tolerance": rel_max is not None and rel_max <= resid_rel_tol,
    }
    return {
        "passed": all(checks.values()) and n_run > 0,
        "checks": checks,
        "tolerances": {"e_abs": e_tol, "residual_max_abs_over_rms": resid_rel_tol},
        "n_requested": n, "n_sampled": len(records), "n_run": n_run, "seed": seed,
        "stage": stage, "cache": str(cache), "run_settings": settings,
        "counts": counts,
        "e_abs_diff": e,
        "e_post_abs_diff": _stats(epost_diffs),
        "residual_abs_diff": {p: {"max": max(v) if v else None,
                                  "mean": float(np.mean(resid_mean_abs[p])) if v else None}
                              for p, v in resid_abs.items()},
        "residual_max_abs_over_rms": {p: (max(v) if v else None) for p, v in resid_rel.items()},
        "positions_compared": list(PROMPT_POSITIONS),
        "positions_not_compared": ["analysis_end", "reasoning_end", "pre_score"],
        "backend": backend.base_meta() if hasattr(backend, "base_meta") else {"backend": backend.name},
        "versions": package_versions(),
        "september_commit": sep.SEPTEMBER_COMMIT,
        "rows": rows,
    }
