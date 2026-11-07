import numpy as np
import pandas as pd
import pytest

from src.config import RAW_FEATURES
from src.features.build_features import (
    AGE_LABELS,
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    FeatureEngineer,
    add_features,
)


def one_client(**overrides):
    row = {c: 0 for c in RAW_FEATURES}
    row.update(LIMIT_BAL=100_000.0, SEX=1, EDUCATION=2, MARRIAGE=1, AGE=33)
    row.update(PAY_0=2, PAY_2=1, PAY_3=-1, PAY_4=0, PAY_5=-2, PAY_6=3)
    row.update({f"BILL_AMT{i}": 10_000.0 * i for i in range(1, 7)})
    row.update({f"PAY_AMT{i}": 1_000.0 for i in range(1, 7)})
    row.update(overrides)
    return pd.DataFrame([row])


def test_payment_history_aggregates():
    f = add_features(one_client()).iloc[0]
    assert f["pay_delay_max"] == 3
    assert f["pay_delay_mean"] == pytest.approx((2 + 1 + 3) / 6)
    assert f["n_months_delayed"] == 3
    assert f["n_months_delayed_2plus"] == 2


def test_amount_ratios():
    f = add_features(one_client()).iloc[0]
    assert f["utilization_last"] == pytest.approx(10_000 / 100_000)
    assert f["utilization_mean"] == pytest.approx(35_000 / 100_000)
    assert f["payment_ratio"] == pytest.approx(6_000 / 210_000)
    assert f["bill_trend"] == pytest.approx((10_000 - 60_000) / 100_000)


@pytest.mark.parametrize("age,label", [(21, "<25"), (25, "25-29"), (39, "35-39"), (59, "50-59"), (75, "60+")])
def test_age_binning(age, label):
    assert add_features(one_client(AGE=age))["AGE_BIN"].iloc[0] == label
    assert label in AGE_LABELS


def test_no_nan_or_inf_on_zero_limit_and_bills():
    zero = one_client(LIMIT_BAL=0.0, **{f"BILL_AMT{i}": 0.0 for i in range(1, 7)})
    f = add_features(zero)[ENGINEERED_NUMERIC].to_numpy(dtype=float)
    assert np.isfinite(f).all()


def test_feature_engineer_transformer(clean_df):
    X = clean_df[RAW_FEATURES]
    out = FeatureEngineer().fit_transform(X)
    assert set(ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL) <= set(out.columns)
    assert len(out) == len(X)
    passthrough = FeatureEngineer(enabled=False).fit_transform(X)
    pd.testing.assert_frame_equal(passthrough, X)
