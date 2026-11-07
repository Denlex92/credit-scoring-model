import numpy as np
import pytest

from src.config import RAW_FEATURES, TARGET
from src.models.metrics import best_f1_threshold, compute_metrics, ks_statistic
from src.models.pipeline import MODELS, create_pipeline


@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("fe", [True, False])
def test_pipeline_fits_and_predicts_probabilities(clean_df, model, fe):
    pipe = create_pipeline(model, feature_engineering=fe)
    pipe.fit(clean_df[RAW_FEATURES], clean_df[TARGET])
    proba = pipe.predict_proba(clean_df[RAW_FEATURES].head(20))
    assert proba.shape == (20, 2)
    assert np.allclose(proba.sum(1), 1)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_handles_unseen_category(clean_df):
    pipe = create_pipeline("logreg").fit(clean_df[RAW_FEATURES], clean_df[TARGET])
    new = clean_df[RAW_FEATURES].head(1).assign(EDUCATION=9)
    assert pipe.predict_proba(new).shape == (1, 2)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        create_pipeline("xgboost-not-there")


def test_metrics_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == 1.0 and m["gini"] == 1.0 and m["ks"] == 1.0
    assert m["precision"] == m["recall"] == m["f1"] == 1.0


def test_metrics_threshold_changes_classes():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.3, 0.4, 0.6, 0.2])
    assert compute_metrics(y, p, 0.5)["recall"] == 0.5
    assert compute_metrics(y, p, 0.35)["recall"] == 1.0


def test_random_scores_have_auc_near_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    m = compute_metrics(y, rng.random(5000))
    assert abs(m["roc_auc"] - 0.5) < 0.03
    assert ks_statistic(y, rng.random(5000)) < 0.06


def test_best_f1_threshold_separates_classes():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8])
    t = best_f1_threshold(y, p)
    assert 0.3 < t <= 0.7
