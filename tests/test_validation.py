"""Тесты набора правил Great Expectations.

Тесты на аномалии проверяют, что валидация ПРОВАЛИВАЕТСЯ на испорченных данных.
test_raw_data_file_is_valid проверяет реальный файл data/raw/UCI_Credit_Card.csv
и упадёт, если в новых данных появятся аномалии (запускается в CI после загрузки данных).
"""

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET, load_params, resolve

pytest.importorskip("great_expectations")
from src.data.validation import validate_dataframe  # noqa: E402


def failed_columns(result):
    return {f["column"] for f in result["failed_expectations"]}


def test_clean_synthetic_data_passes(raw_df):
    result = validate_dataframe(raw_df)
    assert result["success"], result["failed_expectations"]
    assert result["evaluated_expectations"] > 50


def test_nulls_are_detected(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "LIMIT_BAL"] = np.nan
    result = validate_dataframe(bad)
    assert not result["success"]
    assert "LIMIT_BAL" in failed_columns(result)


@pytest.mark.parametrize(
    "column,value",
    [("AGE", 150), ("SEX", 3), ("PAY_0", 15), ("PAY_AMT1", -500.0), (TARGET, 2), ("MARRIAGE", 7)],
)
def test_out_of_range_values_are_detected(raw_df, column, value):
    bad = raw_df.copy()
    bad.loc[0, column] = value
    result = validate_dataframe(bad)
    assert not result["success"]
    assert column in failed_columns(result)


def test_missing_column_is_detected(raw_df):
    result = validate_dataframe(raw_df.drop(columns=["BILL_AMT3"]))
    assert not result["success"]


def test_wrong_type_is_detected(raw_df):
    bad = raw_df.copy()
    bad["AGE"] = bad["AGE"].astype(str)
    assert not validate_dataframe(bad)["success"]


def test_inference_data_without_target(raw_df):
    assert validate_dataframe(raw_df.drop(columns=[TARGET]), require_target=False)["success"]


def test_raw_data_file_is_valid():
    path = resolve(load_params()["data"]["raw_path"])
    if not path.exists():
        pytest.skip("реальные данные не загружены (python -m src.data.download)")
    result = validate_dataframe(pd.read_csv(path))
    assert result["success"], f"Аномалии во входных данных: {result['failed_expectations']}"
