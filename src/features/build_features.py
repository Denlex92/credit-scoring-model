"""Feature Engineering для PD-модели.

Признаки строятся из сырых колонок внутри sklearn-трансформера, поэтому один и тот же код
используется при обучении, в API и в мониторинге (нет расхождения train/serving).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import BILL_COLS, PAY_AMT_COLS, PAY_STATUS_COLS

AGE_BINS = [0, 25, 30, 35, 40, 50, 60, 200]
AGE_LABELS = ["<25", "25-29", "30-34", "35-39", "40-49", "50-59", "60+"]

ENGINEERED_NUMERIC = [
    "pay_delay_max",
    "pay_delay_mean",
    "n_months_delayed",
    "n_months_delayed_2plus",
    "bill_amt_mean",
    "pay_amt_mean",
    "utilization_last",
    "utilization_mean",
    "payment_ratio",
    "bill_trend",
    "limit_per_age",
]
ENGINEERED_CATEGORICAL = ["AGE_BIN"]


def _safe_div(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет агрегированные признаки истории платежей и биннинг возраста."""
    out = df.copy()
    pay = out[PAY_STATUS_COLS].to_numpy(dtype=float)
    delays = np.clip(pay, 0, None)  # -2/-1/0 = нет просрочки
    bills = out[BILL_COLS].to_numpy(dtype=float)
    pays = out[PAY_AMT_COLS].to_numpy(dtype=float)
    limit = out["LIMIT_BAL"].to_numpy(dtype=float)

    out["pay_delay_max"] = delays.max(axis=1)
    out["pay_delay_mean"] = delays.mean(axis=1)
    out["n_months_delayed"] = (pay > 0).sum(axis=1)
    out["n_months_delayed_2plus"] = (pay >= 2).sum(axis=1)
    out["bill_amt_mean"] = bills.mean(axis=1)
    out["pay_amt_mean"] = pays.mean(axis=1)
    out["utilization_last"] = _safe_div(bills[:, 0], limit)
    out["utilization_mean"] = _safe_div(bills.mean(axis=1), limit)
    # доля выставленных счетов, которая была оплачена (ограничиваем выбросы)
    out["payment_ratio"] = np.clip(_safe_div(pays.sum(axis=1), np.clip(bills.sum(axis=1), 1, None)), 0, 5)
    out["bill_trend"] = _safe_div(bills[:, 0] - bills[:, 5], limit)
    out["limit_per_age"] = _safe_div(limit, out["AGE"])
    out["AGE_BIN"] = pd.cut(out["AGE"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """sklearn-обёртка над add_features. enabled=False пропускает данные без изменений."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return add_features(X) if self.enabled else X
