"""Метрики дрифта данных: Population Stability Index (PSI)."""

import numpy as np


def psi(expected, actual, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI между эталонным (expected) и текущим (actual) распределением.

    Границы бинов — квантили эталонного распределения (для дискретных признаков
    повторяющиеся границы схлопываются). PSI = Σ (a_i − e_i) · ln(a_i / e_i).

    Интерпретация: < 0.1 — стабильно, 0.1–0.25 — умеренный сдвиг, > 0.25 — значимый дрифт.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # константный признак
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value: float, warning: float = 0.1, alert: float = 0.25) -> str:
    if value >= alert:
        return "ALERT"
    if value >= warning:
        return "WARNING"
    return "OK"
