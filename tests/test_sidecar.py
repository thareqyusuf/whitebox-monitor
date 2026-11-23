import numpy as np
import pytest

from whitebox_monitor.store import SidecarReader, SidecarWriter
from whitebox_monitor.store.sidecar import iter_index


def _residuals(rng, d=16):
    return {(layer, pos): rng.standard_normal(d).astype(np.float32)
            for layer in (0, 3) for pos in ("action_last", "action_end")}


def test_round_trip_identical_arrays_and_keys(tmp_path):
    rng = np.random.default_rng(0)
    written = {}
    with SidecarWriter(tmp_path, shard_size=5) as w:
        for sample in range(3):
            for msg in ("m1", "m2"):
                res = _residuals(rng)
                ref = w.add("run-a", sample, msg, res)
                assert ref["dtype"] == "float16" and len(ref["keys"]) == 4
                for (layer, pos), vec in res.items():
                    written[("run-a", str(sample), msg, layer, pos)] = vec.astype(np.float16)
    r = SidecarReader(tmp_path)
    assert r.keys() == sorted(written) and len(r) == 24
    for key, vec in written.items():
        got = r.get(*key)
        assert got.dtype == np.float16 and np.array_equal(got, vec)
    action = r.get_action("run-a", 1, "m2")
    assert sorted(action) == [(0, "action_end"), (0, "action_last"), (3, "action_end"), (3, "action_last")]
    rows = list(iter_index(tmp_path))
    assert all(row["dtype"] == "float16" and row["source_dtype"] == "float32" for row in rows)
    assert len({row["shard"] for row in rows}) > 1


def test_float32_store_is_exact(tmp_path):
    rng = np.random.default_rng(1)
    res = _residuals(rng)
    with SidecarWriter(tmp_path, dtype="float32") as w:
        w.add("r", "s", "m", res)
    r = SidecarReader(tmp_path)
    for (layer, pos), vec in res.items():
        assert np.array_equal(r.get("r", "s", "m", layer, pos), vec)


def test_duplicates_and_bad_values_raise(tmp_path):
    with SidecarWriter(tmp_path) as w:
        w.add("r", "s", "m", {(0, "action_last"): np.ones(4)})
        with pytest.raises(KeyError):
            w.add("r", "s", "m", {(0, "action_last"): np.ones(4)})
        with pytest.raises(ValueError, match="non-finite"):
            w.add("r", "s", "m2", {(0, "action_last"): np.array([np.nan])})
        with pytest.raises(ValueError, match="overflows"):
            w.add("r", "s", "m3", {(0, "action_last"): np.array([1e6])})
    # a second writer on the same directory also refuses the key
    with SidecarWriter(tmp_path) as w2, pytest.raises(KeyError):
        w2.add("r", "s", "m", {(0, "action_last"): np.ones(4)})


def test_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        SidecarReader(tmp_path)
    with SidecarWriter(tmp_path) as w:
        w.add("r", "s", "m", {(0, "action_last"): np.ones(4)})
    with pytest.raises(KeyError):
        SidecarReader(tmp_path).get("r", "s", "m", 1, "action_last")
