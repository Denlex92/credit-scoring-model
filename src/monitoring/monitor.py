"""Имитация потока новых данных и мониторинг дрифта.

Скрипт берёт случайную часть тестовой выборки («новые заявки»), отправляет каждую запись
в REST API (/predict), собирает вероятности дефолта и считает PSI относительно
тренировочных данных: для распределения вероятностей и для ключевых признаков.
Флаг --shift искусственно сдвигает данные, чтобы показать срабатывание алерта.

    python -m src.monitoring.monitor                       # через API на localhost:8000
    python -m src.monitoring.monitor --local               # без API, модель из models/
    python -m src.monitoring.monitor --shift 0.5           # имитация дрифта
"""

import argparse
import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.config import RAW_FEATURES, TARGET, load_params, resolve  # noqa: E402
from src.models.predict import load_model  # noqa: E402
from src.monitoring.drift import psi, psi_status  # noqa: E402


def simulate_shift(df: pd.DataFrame, strength: float, rng) -> pd.DataFrame:
    """Сдвиг «экономического кризиса»: лимиты ниже, просрочки чаще, платежи меньше."""
    out = df.copy()
    out["LIMIT_BAL"] = (out["LIMIT_BAL"] * (1 - 0.5 * strength)).clip(lower=10_000).round()
    worsen = rng.random(len(out)) < strength
    out.loc[worsen, "PAY_0"] = np.clip(out.loc[worsen, "PAY_0"] + 2, -2, 8)
    pay_amt = [c for c in RAW_FEATURES if c.startswith("PAY_AMT")]
    out[pay_amt] = (out[pay_amt] * (1 - 0.6 * strength)).round()
    return out


def predict_via_api(df: pd.DataFrame, url: str) -> np.ndarray:
    probs = []
    with requests.Session() as s:
        for rec in df[RAW_FEATURES].to_dict(orient="records"):
            payload = {k: (int(v) if float(v).is_integer() else float(v)) for k, v in rec.items()}
            r = s.post(f"{url}/predict", json=payload, timeout=10)
            r.raise_for_status()
            probs.append(r.json()["default_probability"])
    return np.array(probs)


def main() -> None:
    params = load_params()
    mcfg, tcfg = params["monitoring"], params["train"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=mcfg["api_url"])
    parser.add_argument("--n", type=int, default=mcfg["n_samples"])
    parser.add_argument("--shift", type=float, default=0.0, help="сила искусственного дрифта 0..1")
    parser.add_argument("--local", action="store_true", help="считать предсказания локально, без API")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    processed = resolve(params["data"]["processed_dir"])
    reference = pd.read_csv(processed / "train.csv")
    current = pd.read_csv(processed / "test.csv").sample(n=args.n, random_state=args.seed)
    if args.shift > 0:
        current = simulate_shift(current, args.shift, rng)

    model, meta = load_model(tcfg["model_path"], tcfg["meta_path"])
    ref_proba = model.predict_proba(reference[RAW_FEATURES])[:, 1]
    if args.local:
        cur_proba = model.predict_proba(current[RAW_FEATURES])[:, 1]
        source = "local model"
    else:
        cur_proba = predict_via_api(current, args.url)
        source = f"API {args.url}"

    rows = [("default_probability", psi(ref_proba, cur_proba, mcfg["psi_bins"]))]
    rows += [(f, psi(reference[f], current[f], mcfg["psi_bins"])) for f in mcfg["features"]]
    table = pd.DataFrame(rows, columns=["variable", "psi"])
    table["status"] = [psi_status(v, mcfg["psi_warning"], mcfg["psi_alert"]) for v in table["psi"]]

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "n_samples": int(len(current)),
        "shift": args.shift,
        "mean_probability_reference": float(ref_proba.mean()),
        "mean_probability_current": float(cur_proba.mean()),
        "predicted_default_rate": float((cur_proba >= meta["threshold"]).mean()),
        "roc_auc_current": float(roc_auc_score(current[TARGET], cur_proba)) if current[TARGET].nunique() > 1 else None,
        "psi": table.to_dict(orient="records"),
        "overall_status": (
            "ALERT" if (table.status == "ALERT").any() else "WARNING" if (table.status == "WARNING").any() else "OK"
        ),
    }
    out = resolve(mcfg["report_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(0, 1, 31)
    axes[0].hist(ref_proba, bins=bins, density=True, alpha=0.6, label="train (эталон)")
    axes[0].hist(cur_proba, bins=bins, density=True, alpha=0.6, label="новые данные")
    axes[0].set(title="Распределение вероятности дефолта", xlabel="P(default)", ylabel="Плотность")
    axes[0].legend()
    colors = {"OK": "#55A868", "WARNING": "#DD8452", "ALERT": "#C44E52"}
    axes[1].barh(table["variable"], table["psi"], color=[colors[s] for s in table["status"]])
    for x, ls in [(mcfg["psi_warning"], ":"), (mcfg["psi_alert"], "--")]:
        axes[1].axvline(x, color="grey", ls=ls)
    axes[1].set(title="PSI по переменным (пунктир: 0.1 и 0.25)", xlabel="PSI")
    axes[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(resolve("reports/figures/drift_report.png"), dpi=120)

    print(f"Источник предсказаний: {source}, записей: {len(current)}")
    print(table.round(4).to_string(index=False))
    print(
        f"Средняя P(default): train {report['mean_probability_reference']:.3f} -> "
        f"новые {report['mean_probability_current']:.3f}; ROC-AUC на новых: {report['roc_auc_current']}"
    )
    print(f"Итоговый статус: {report['overall_status']} | отчёт: {out}")


if __name__ == "__main__":
    main()
