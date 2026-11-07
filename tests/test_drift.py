import numpy as np

from src.monitoring.drift import psi, psi_status


def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=10_000), rng.normal(size=10_000)) < 0.02


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=10_000), rng.normal(1.0, 1, size=10_000)) > 0.25


def test_psi_discrete_feature_and_constant():
    rng = np.random.default_rng(0)
    ref = rng.choice([-2, -1, 0, 1, 2], 5000, p=[0.1, 0.2, 0.5, 0.1, 0.1])
    assert psi(ref, ref) == 0.0
    assert psi(np.ones(100), np.ones(100) * 5) == 0.0


def test_psi_status_thresholds():
    assert psi_status(0.05) == "OK"
    assert psi_status(0.15) == "WARNING"
    assert psi_status(0.3) == "ALERT"
