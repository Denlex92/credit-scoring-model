import numpy as np
import pandas as pd
import pytest

from src.config import BILL_COLS, PAY_AMT_COLS, PAY_STATUS_COLS, TARGET


def make_credit_df(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Синтетические данные в формате UCI_Credit_Card.csv (для тестов без реального датасета)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "ID": np.arange(1, n + 1),
            "LIMIT_BAL": rng.choice([20_000, 50_000, 100_000, 200_000, 360_000], n).astype(float),
            "SEX": rng.integers(1, 3, n),
            "EDUCATION": rng.choice([0, 1, 2, 3, 4, 5, 6], n, p=[0.01, 0.35, 0.46, 0.15, 0.01, 0.01, 0.01]),
            "MARRIAGE": rng.choice([0, 1, 2, 3], n, p=[0.01, 0.45, 0.53, 0.01]),
            "AGE": rng.integers(21, 75, n),
        }
    )
    risk = rng.normal(0, 1, n)
    for c in PAY_STATUS_COLS:
        df[c] = np.clip(np.round(risk + rng.normal(0, 0.8, n)).astype(int), -2, 8)
    for c in BILL_COLS:
        df[c] = np.round(df["LIMIT_BAL"] * rng.uniform(-0.05, 0.9, n))
    for c in PAY_AMT_COLS:
        df[c] = np.round(rng.exponential(3000, n) * (risk < 1))
    p = 1 / (1 + np.exp(-(-1.4 + 1.2 * risk)))
    df[TARGET] = (rng.random(n) < p).astype(int)
    return df


@pytest.fixture
def raw_df():
    return make_credit_df()


@pytest.fixture
def clean_df(raw_df):
    from src.data.make_dataset import clean_data

    return clean_data(raw_df)
