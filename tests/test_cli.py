"""`wbm run` end to end on a tiny model saved to disk (CPU, no downloads)."""

import json

import numpy as np
import pytest
from conftest import TEMPLATE

from whitebox_monitor.cli import main, prompt_from_json
from whitebox_monitor.store import SidecarReader


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, tiny_model, tokenizer):
    path = tmp_path_factory.mktemp("tiny-model")
    tiny_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def _write_prompts(path):
    action = "Tool call: bash: {'cmd': 'cat /etc/passwd'}"
    text = TEMPLATE.format(history="No previous actions.", action=action)
    start = text.index("<current_action>\n") + len("<current_action>\n")
    rows = [
        {"id": "explicit", "messages": [{"role": "user", "content": text}],
         "spans": {"action": [start, start + len(action)]}},
        {"id": "templated", "template": TEMPLATE,
         "fields": {"history": "No previous actions.", "action": action},
         "extra_span_fields": {"history": "history"}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return rows


def test_run(tmp_path, model_dir, capsys):
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts)
    out = tmp_path / "out"
    code = main(["run", "--prompts", str(prompts), "--out", str(out), "--model", str(model_dir),
                 "--layers", "0,2,3", "--positions", "action_last,action_end",
                 "--max-new-tokens", "8", "--run-id", "t1"])
    assert code == 0
    rows = [json.loads(line) for line in (out / "readouts.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["explicit", "templated"]
    reader = SidecarReader(out / "acts")
    assert len(reader) == 2 * 3 * 2
    for r in rows:
        assert r["status"] == "ok" and r["integer"] is None       # random model: no score tag
        assert r["meta"]["integer_parse"]["reason"]
        assert 0 <= r["expected"] <= 10
        assert sum(r["score_token_probs"].values()) == pytest.approx(1.0)
        assert r["action_token_index"] == r["meta"]["positions"]["action_last"]["token_index"]
        assert r["meta"]["model_id"] == str(model_dir) and r["meta"]["backend"] == "hf_hooks"
        assert r["meta"]["versions"]["whitebox-monitor"]
        vec = reader.get("t1", r["id"], "prompt", 3, "action_last")
        assert vec.dtype == np.float16 and vec.shape == (32,)
    # the two prompt forms describe the same prompt, so they give the same numbers
    assert rows[0]["expected"] == rows[1]["expected"]
    assert np.array_equal(reader.get("t1", "explicit", "prompt", 2, "action_end"),
                          reader.get("t1", "templated", "prompt", 2, "action_end"))
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["n_failed"] == 0 and meta["capture"]["layers"] == [0, 2, 3]
    # refuses to clobber
    with pytest.raises(SystemExit):
        main(["run", "--prompts", str(prompts), "--out", str(out), "--model", str(model_dir)])


def test_run_records_failures_and_exits_nonzero(tmp_path, model_dir):
    prompts = tmp_path / "prompts.jsonl"
    rows = _write_prompts(prompts)
    rows[0]["spans"] = {"history": [0, 3]}                       # no action span
    prompts.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "out"
    code = main(["run", "--prompts", str(prompts), "--out", str(out), "--model", str(model_dir),
                 "--layers", "1", "--no-generate"])
    assert code == 1
    got = [json.loads(line) for line in (out / "readouts.jsonl").read_text().splitlines()]
    assert got[0]["status"] == "failed" and "action" in got[0]["error"]
    assert got[1]["status"] == "ok" and got[1]["integer"] is None


def test_missing_inputs(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        main(["run", "--prompts", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path / "o"),
              "--model", "x"])
    with pytest.raises(SystemExit, match="cache"):
        main(["validate-september", "--cache", str(tmp_path / "no-cache"),
              "--goals", str(tmp_path / "goals.json")])
    with pytest.raises(ValueError):
        prompt_from_json({"id": 1})


def test_info(capsys):
    assert main(["info"]) == 0
    assert "whitebox-monitor" in capsys.readouterr().out
