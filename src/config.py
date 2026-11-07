"""Общие константы проекта и загрузка параметров из params.yaml."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"

TARGET = "default.payment.next.month"
ID_COL = "ID"

PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
CATEGORICAL_RAW = ["SEX", "EDUCATION", "MARRIAGE"]

# 23 исходных признака датасета в порядке UCI
RAW_FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY_STATUS_COLS + BILL_COLS + PAY_AMT_COLS


def load_params(path: Path = PARAMS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str) -> Path:
    """Путь относительно корня репозитория."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
