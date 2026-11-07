import importlib
import json

import joblib
import pytest

from src.config import RAW_FEATURES, TARGET
from src.models.pipeline import create_pipeline

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch, clean_df):
    model = create_pipeline("logreg").fit(clean_df[RAW_FEATURES], clean_df[TARGET])
    joblib.dump(model, tmp_path / "model.pkl")
    (tmp_path / "meta.json").write_text(json.dumps({"threshold": 0.4, "model_name": "test_model"}))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.pkl"))
    monkeypatch.setenv("META_PATH", str(tmp_path / "meta.json"))
    import src.api.app as app_module

    app_module = importlib.reload(app_module)
    return TestClient(app_module.app), app_module.ClientData.model_config["json_schema_extra"]["example"]


def test_root(client):
    c, _ = client
    assert c.get("/").status_code == 200


def test_health(client):
    c, _ = client
    body = c.get("/health").json()
    assert body["status"] == "ok" and body["model_name"] == "test_model"


def test_predict_returns_class_and_probability(client):
    c, example = client
    resp = c.post("/predict", json=example)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["default_prediction"] == int(body["default_probability"] >= 0.4)


@pytest.mark.parametrize("field,value", [("AGE", 10), ("SEX", 5), ("PAY_AMT1", -1)])
def test_predict_rejects_invalid_input(client, field, value):
    c, example = client
    assert c.post("/predict", json={**example, field: value}).status_code == 422


def test_predict_rejects_missing_field(client):
    c, example = client
    payload = {k: v for k, v in example.items() if k != "LIMIT_BAL"}
    assert c.post("/predict", json=payload).status_code == 422
