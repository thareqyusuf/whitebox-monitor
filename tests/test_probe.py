import numpy as np
import pytest

from whitebox_monitor import LinearProbe, load_probe
from whitebox_monitor.core.probe import save_probe


def _probe(**kw):
    base = dict(weights=np.array([1.0, -2.0, 0.5]), bias=0.25, layer=2,
                position="action_last", manifest_sha256="ab" * 32)
    base.update(kw)
    return LinearProbe(**base)


def test_apply_matches_hand_computation():
    p = _probe()
    x = np.array([2.0, 1.0, 4.0], dtype=np.float16)
    assert p.apply({(2, "action_last"): x}) == pytest.approx(2 - 2 + 2 + 0.25)


def test_scaler_is_applied():
    p = _probe(scaler_mean=np.array([1.0, 1.0, 0.0]), scaler_scale=np.array([2.0, 1.0, 4.0]))
    assert p.apply_vector(np.array([3.0, 1.0, 8.0])) == pytest.approx(1 + 0 + 1 + 0.25)


def test_round_trip(tmp_path):
    p = _probe(scaler_mean=np.zeros(3), scaler_scale=np.ones(3))
    path = tmp_path / "probe.npz"
    save_probe(path, p)
    q = load_probe(path)
    assert q.layer == 2 and q.position == "action_last" and q.manifest_sha256 == "ab" * 32
    assert np.array_equal(q.weights, p.weights) and q.bias == p.bias
    assert len(q.file_sha256) == 64 and q.to_meta()["has_scaler"]
    x = np.array([0.3, 0.1, -0.7])
    assert q.apply_vector(x) == p.apply_vector(x)


def test_failures_are_loud(tmp_path):
    p = _probe()
    with pytest.raises(KeyError):
        p.apply({(1, "action_last"): np.zeros(3)})
    with pytest.raises(ValueError, match="shape"):
        p.apply_vector(np.zeros(4))
    with pytest.raises(ValueError, match="non-finite"):
        p.apply_vector(np.array([np.nan, 0, 0]))
    with pytest.raises(FileNotFoundError):
        load_probe(tmp_path / "missing.npz")
    np.savez(tmp_path / "bad.npz", weights=np.zeros(3))
    with pytest.raises(ValueError, match="missing fields"):
        load_probe(tmp_path / "bad.npz")
    with pytest.raises(ValueError):
        _probe(scaler_mean=np.zeros(3))
