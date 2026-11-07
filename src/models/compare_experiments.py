"""Сравнение экспериментов из MLflow: таблица и график по всем run'ам эксперимента.

python -m src.models.compare_experiments
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

from src.config import load_params, resolve  # noqa: E402

COLS = {
    "tags.mlflow.runName": "run",
    "params.model": "model",
    "params.feature_engineering": "FE",
    "metrics.cv_roc_auc": "cv_roc_auc",
    "metrics.cv_roc_auc_std": "cv_std",
    "metrics.test_roc_auc": "test_roc_auc",
    "metrics.test_gini": "test_gini",
    "metrics.test_ks": "test_ks",
    "metrics.test_precision": "precision",
    "metrics.test_recall": "recall",
    "metrics.test_f1": "f1",
    "metrics.threshold": "threshold",
    "metrics.fit_time_sec": "fit_time_sec",
}


def main() -> None:
    cfg = load_params()["train"]
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    runs = mlflow.search_runs(experiment_names=[cfg["experiment_name"]], order_by=["metrics.cv_roc_auc DESC"])
    if runs.empty:
        print("Нет run'ов — сначала запустите обучение")
        return
    table = runs[[c for c in COLS if c in runs.columns]].rename(columns=COLS)
    table = table.drop_duplicates("run", keep="first")  # последний запуск каждого эксперимента
    out = resolve("reports")
    out.mkdir(exist_ok=True)
    table.to_csv(out / "experiments_comparison.csv", index=False)
    (out / "experiments_comparison.md").write_text(table.round(4).to_markdown(index=False), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(table) + 1.5))
    ax.barh(table["run"], table["cv_roc_auc"], xerr=table["cv_std"], color="#4C72B0", label="CV ROC-AUC")
    ax.scatter(table["test_roc_auc"], table["run"], color="#C44E52", zorder=3, label="test ROC-AUC")
    ax.set(
        xlim=(table["cv_roc_auc"].min() - 0.03, table[["cv_roc_auc", "test_roc_auc"]].max().max() + 0.01),
        xlabel="ROC-AUC",
        title="Сравнение экспериментов (MLflow)",
    )
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "figures" / "experiments_comparison.png", dpi=120)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
