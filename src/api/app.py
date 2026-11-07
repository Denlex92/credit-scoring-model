"""REST API для PD-модели (FastAPI).

    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Эндпоинты:
    GET  /         — проверка, что сервис жив
    GET  /health   — статус и информация о загруженной модели
    POST /predict  — признаки клиента (JSON) -> класс и вероятность дефолта
"""

import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import RAW_FEATURES, load_params
from src.models.predict import load_model

_cfg = load_params()["train"]
MODEL_PATH = os.getenv("MODEL_PATH", _cfg["model_path"])
META_PATH = os.getenv("META_PATH", _cfg["meta_path"])

app = FastAPI(title="Credit Default Prediction API", version="1.0.0")


class ClientData(BaseModel):
    LIMIT_BAL: float = Field(..., gt=0, description="Кредитный лимит, NT$")
    SEX: int = Field(..., ge=1, le=2, description="1 = мужчина, 2 = женщина")
    EDUCATION: int = Field(..., ge=0, le=6, description="1 = магистратура, 2 = университет, 3 = школа, 4 = другое")
    MARRIAGE: int = Field(..., ge=0, le=3, description="1 = в браке, 2 = холост, 3 = другое")
    AGE: int = Field(..., ge=18, le=100)
    PAY_0: int = Field(
        ..., ge=-2, le=9, description="Статус платежа в сентябре (-1 = вовремя, 1..9 = месяцев просрочки)"
    )
    PAY_2: int = Field(..., ge=-2, le=9)
    PAY_3: int = Field(..., ge=-2, le=9)
    PAY_4: int = Field(..., ge=-2, le=9)
    PAY_5: int = Field(..., ge=-2, le=9)
    PAY_6: int = Field(..., ge=-2, le=9)
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(..., ge=0)
    PAY_AMT2: float = Field(..., ge=0)
    PAY_AMT3: float = Field(..., ge=0)
    PAY_AMT4: float = Field(..., ge=0)
    PAY_AMT5: float = Field(..., ge=0)
    PAY_AMT6: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
                "PAY_0": 2, "PAY_2": 2, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 46990, "BILL_AMT2": 48233, "BILL_AMT3": 49291,
                "BILL_AMT4": 28314, "BILL_AMT5": 28959, "BILL_AMT6": 29547,
                "PAY_AMT1": 2000, "PAY_AMT2": 2019, "PAY_AMT3": 1200,
                "PAY_AMT4": 1100, "PAY_AMT5": 1069, "PAY_AMT6": 1000,
            }
        }
    }  # fmt: skip


class Prediction(BaseModel):
    default_prediction: int
    default_probability: float
    threshold: float
    model_name: str


def get_model():
    try:
        return load_model(MODEL_PATH, META_PATH)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Модель не найдена: {e}") from e


@app.get("/")
def read_root():
    return {"message": "Credit Default Prediction API is alive!"}


@app.get("/health")
def health():
    _, meta = get_model()
    return {"status": "ok", "model_name": meta.get("model_name"), "threshold": meta.get("threshold")}


@app.post("/predict", response_model=Prediction)
def predict(data: ClientData):
    model, meta = get_model()
    X = pd.DataFrame([data.model_dump()])[RAW_FEATURES]
    proba = float(model.predict_proba(X)[0, 1])
    threshold = float(meta.get("threshold", 0.5))
    return Prediction(
        default_prediction=int(proba >= threshold),
        default_probability=proba,
        threshold=threshold,
        model_name=str(meta.get("model_name", "unknown")),
    )
