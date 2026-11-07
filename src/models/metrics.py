"""Метрики качества бинарной классификации и графики для PD-модели."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true, y_proba) -> float:
    """Статистика Колмогорова–Смирнова: max(TPR - FPR), стандартная метрика скоринга."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_proba)
    return {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "ks": ks_statistic(y_true, y_proba),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def best_f1_threshold(y_true, y_proba) -> float:
    """Порог, максимизирующий F1 (подбирается на out-of-fold предсказаниях train)."""
    precision, recall, thr = precision_recall_curve(y_true, y_proba)
    f1 = np.divide(
        2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0
    )
    return float(thr[np.argmax(f1[:-1])])


def plot_roc_curve(y_true, y_proba, path, title: str = "ROC-кривая"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="случайная модель")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate (Recall)", title=title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion_matrix(y_true, y_proba, threshold, path):
    cm = confusion_matrix(y_true, (np.asarray(y_proba) >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["нет дефолта", "дефолт"],
        yticklabels=["нет дефолта", "дефолт"],
        xlabel="Предсказание",
        ylabel="Факт",
        title=f"Матрица ошибок (порог {threshold:.2f})",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_feature_importance(names, values, path, top: int = 20, title: str = "Важность признаков"):
    order = np.argsort(np.abs(values))[::-1][:top]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(order) + 1.5))
    ax.barh(np.asarray(names)[order][::-1], np.asarray(values)[order][::-1], color="#4C72B0")
    ax.set(title=title, xlabel="Важность")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
