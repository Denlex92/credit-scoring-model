"""Стадия train: подбор гиперпараметров, логирование экспериментов в MLflow, сохранение лучшей модели.

Каждый эксперимент из params.yaml (алгоритм + Feature Engineering + сетка) — отдельный run в MLflow:
параметры, CV- и тестовые метрики, графики (ROC, матрица ошибок, важность признаков),
таблица результатов поиска и сама модель (mlflow.sklearn.log_model).
Лучшая по CV ROC-AUC модель сохраняется в models/ и регистрируется в MLflow Model Registry.

    python -m src.models.train                    # все эксперименты
    python -m src.models.train --only logreg_fe   # выборочно (например, в CI)
"""

import argparse
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)

from src.config import RAW_FEATURES, TARGET, load_params, resolve
from src.models.metrics import (
    best_f1_threshold,
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)
from src.models.pipeline import create_pipeline


def load_split(processed_dir):
    d = resolve(processed_dir)
    train, test = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
    return train[RAW_FEATURES], train[TARGET], test[RAW_FEATURES], test[TARGET]


def make_search(exp: dict, cfg: dict):
    pipe = create_pipeline(exp["model"], exp["feature_engineering"], cfg["random_state"])
    grid = {f"classifier__{k}": v for k, v in exp["params"].items()}
    cv = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=cfg["random_state"])
    common = dict(scoring=cfg["scoring"], cv=cv, n_jobs=-1, refit=True, return_train_score=False)
    if exp.get("search", "grid") == "random":
        n_iter = min(exp.get("n_iter", 10), len(ParameterGrid(grid)))
        return RandomizedSearchCV(pipe, grid, n_iter=n_iter, random_state=cfg["random_state"], **common), cv
    return GridSearchCV(pipe, grid, **common), cv


def feature_importance(pipeline):
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "coef_"):
        return names, clf.coef_.ravel(), "Коэффициенты логистической регрессии (стандартизованные признаки)"
    if hasattr(clf, "feature_importances_"):
        return names, clf.feature_importances_, "Важность признаков (impurity-based)"
    return None


def run_experiment(exp: dict, cfg: dict, data) -> dict:
    X_train, y_train, X_test, y_test = data
    search, cv = make_search(exp, cfg)
    with mlflow.start_run(run_name=exp["name"]) as run:
        t0 = time.time()
        mlflow.set_tags({"model_type": exp["model"], "search": exp.get("search", "grid")})
        mlflow.log_params(
            {
                "model": exp["model"],
                "feature_engineering": exp["feature_engineering"],
                "search": exp.get("search", "grid"),
                "cv_folds": cfg["cv_folds"],
                "n_candidates": (
                    len(ParameterGrid(search.param_distributions))
                    if hasattr(search, "param_distributions")
                    else len(ParameterGrid(search.param_grid))
                ),
                "n_train": len(X_train),
            }
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        mlflow.log_params({k.replace("classifier__", ""): v for k, v in search.best_params_.items()})

        # порог классификации подбираем по out-of-fold предсказаниям train, тест не трогаем
        oof = cross_val_predict(clone(best), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        threshold = best_f1_threshold(y_train, oof)
        proba = best.predict_proba(X_test)[:, 1]
        m_default = compute_metrics(y_test, proba, 0.5)
        m_tuned = compute_metrics(y_test, proba, threshold)

        cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
        metrics = {"cv_roc_auc": float(search.best_score_), "cv_roc_auc_std": cv_std}
        metrics.update({f"test_{k}": m_default[k] for k in ["roc_auc", "gini", "pr_auc", "ks", "accuracy"]})
        metrics.update({f"test_{k}_at_0.5": m_default[k] for k in ["precision", "recall", "f1"]})
        metrics.update({f"test_{k}": m_tuned[k] for k in ["precision", "recall", "f1"]})
        metrics["threshold"] = threshold
        metrics["fit_time_sec"] = time.time() - t0
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plot_roc_curve(y_test, proba, tmp / "roc_curve.png", f"ROC-кривая: {exp['name']}")
            plot_confusion_matrix(y_test, proba, threshold, tmp / "confusion_matrix.png")
            fi = feature_importance(best)
            if fi is not None:
                plot_feature_importance(*fi[:2], tmp / "feature_importance.png", title=fi[2])
            pd.DataFrame(search.cv_results_).to_csv(tmp / "cv_results.csv", index=False)
            mlflow.log_artifacts(str(tmp), artifact_path="plots")

        signature = infer_signature(X_test.head(50), proba[:50])
        mlflow.sklearn.log_model(
            best,
            "model",
            signature=signature,
            input_example=X_test.head(3),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        print(
            f"[{exp['name']}] CV ROC-AUC = {search.best_score_:.4f} ± {cv_std:.4f} | "
            f"test ROC-AUC = {m_default['roc_auc']:.4f}, F1 = {m_tuned['f1']:.4f} (порог {threshold:.2f}) | "
            f"{metrics['fit_time_sec']:.0f} c"
        )
        return {"name": exp["name"], "run_id": run.info.run_id, "estimator": best, "proba": proba, **metrics}


def main() -> None:
    params = load_params()
    cfg = params["train"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="имена экспериментов через запятую")
    args = parser.parse_args()

    experiments = cfg["experiments"]
    if args.only:
        keep = set(args.only.split(","))
        experiments = [e for e in experiments if e["name"] in keep]
    data = load_split(params["data"]["processed_dir"])

    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])
    results = [run_experiment(exp, cfg, data) for exp in experiments]

    # выбор лучшей модели — по CV (тестовая выборка в выборе не участвует)
    best = max(results, key=lambda r: r["cv_roc_auc"])
    model_path, meta_path = resolve(cfg["model_path"]), resolve(cfg["meta_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["estimator"], model_path)

    test_metrics = {k.replace("test_", ""): v for k, v in best.items() if k.startswith("test_")}
    meta = {
        "model_name": best["name"],
        "mlflow_run_id": best["run_id"],
        "threshold": best["threshold"],
        "features": list(data[0].columns),
        "cv_roc_auc": best["cv_roc_auc"],
        "test_metrics": test_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics_path = resolve(cfg["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps({"model": best["name"], "cv_roc_auc": best["cv_roc_auc"], **test_metrics}, indent=2),
        encoding="utf-8",
    )
    fig_dir = resolve("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    y_test = data[3]
    plot_roc_curve(y_test, best["proba"], fig_dir / "roc_curve.png", f"ROC-кривая лучшей модели: {best['name']}")
    plot_confusion_matrix(y_test, best["proba"], best["threshold"], fig_dir / "confusion_matrix.png")

    summary = pd.DataFrame(
        [
            {k: r[k] for k in ["name", "cv_roc_auc", "test_roc_auc", "test_gini", "test_f1", "threshold"]}
            for r in results
        ]
    ).sort_values("cv_roc_auc", ascending=False)
    summary.to_csv(resolve("reports/experiments_summary.csv"), index=False)
    print("\n" + summary.round(4).to_string(index=False))
    print(f"\nЛучшая модель: {best['name']} -> {model_path}")

    try:
        mlflow.register_model(f"runs:/{best['run_id']}/model", cfg["registered_model_name"])
    except Exception as e:  # реестр недоступен (например, file-store) — не критично
        print(f"Model Registry: модель не зарегистрирована ({e})")


if __name__ == "__main__":
    main()
