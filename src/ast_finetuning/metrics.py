"""Classification metrics computed on validation predictions and exported as CSV."""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def macro_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, float]:
    """Macro-averaged precision, recall and F1: the unweighted mean of the per-class scores.

    Every class is included, so a class absent from the predictions contributes zero.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(num_classes), average="macro", zero_division=0
    )
    return {"precision_macro": float(precision), "recall_macro": float(recall), "f1_macro": float(f1)}


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> pl.DataFrame:
    """Precision, recall, F1 and support for each class."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), average=None, zero_division=0
    )
    return pl.DataFrame(
        {
            "class": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def normalized_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> pl.DataFrame:
    """Confusion matrix normalized by row: cell (i, j) is P(predicted = j | true = i)."""
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)), normalize="true")
    return pl.DataFrame({"true_label": class_names}).hstack(
        pl.DataFrame(matrix, schema=class_names, orient="row")
    )


def save_best_epoch_metrics(
    output_dir: Path,
    epoch: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> None:
    """Write the per-class metrics and normalized confusion matrix of the best epoch."""
    per_class = per_class_metrics(y_true, y_pred, class_names)
    per_class.insert_column(0, pl.Series("epoch", [epoch] * per_class.height))
    per_class.write_csv(output_dir / "per_class_metrics.csv")
    normalized_confusion_matrix(y_true, y_pred, class_names).write_csv(output_dir / "confusion_matrix.csv")
