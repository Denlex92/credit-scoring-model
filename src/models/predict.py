"""Инференс: загрузка сохранённой модели и предсказание вероятности дефолта.

python -m src.models.predict data/processed/test.csv --out reports/predictions.csv
"""

import argparse
import json
from functools import lru_cache

import joblib
import pandas as pd

from src.config import RAW_FEATURES, load_params, resolve


@lru_cache(maxsize=4)
def load_model(model_path: str, meta_path: str):
    model = joblib.load(resolve(model_path))
    meta_file = resolve(meta_path)
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {"threshold": 0.5}
    return model, meta


def predict_df(model, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    proba = model.predict_proba(df[RAW_FEATURES])[:, 1]
    return pd.DataFrame({"default_probability": proba, "default_prediction": (proba >= threshold).astype(int)})


def main() -> None:
    cfg = load_params()["train"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--out", default="reports/predictions.csv")
    args = parser.parse_args()

    model, meta = load_model(cfg["model_path"], cfg["meta_path"])
    preds = predict_df(model, pd.read_csv(resolve(args.input)), meta["threshold"])
    preds.to_csv(resolve(args.out), index=False)
    print(f"{len(preds)} предсказаний -> {args.out}; доля прогнозных дефолтов {preds.default_prediction.mean():.3f}")


if __name__ == "__main__":
    main()
