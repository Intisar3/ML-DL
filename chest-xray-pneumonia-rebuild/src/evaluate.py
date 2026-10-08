"""Evaluation on the untouched test set, with thresholds chosen on validation.

    python -m src.evaluate --data-root /path/to/chest_xray --out outputs
"""
import argparse
import json
from pathlib import Path

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)

from . import model as _model  # noqa: F401  (registers the custom preprocessing layer)
from .config import TASKS, Config
from .data import build_index, make_dataset, make_splits, task_frame

NOTE = "Educational model output. Diagnosis and treatment decisions belong to a clinician."


def predict(model, frame, task, cfg):
    p = model.predict(make_dataset(frame, task, cfg, training=False), verbose=0)
    return p.ravel() if p.shape[-1] == 1 else p


def pick_threshold(y, p) -> float:
    """Youden's J on validation: the point that best balances sensitivity and specificity."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 0.0, 1.0))


def binary_metrics(y, p, thr) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, p >= thr, labels=[0, 1]).ravel()
    return {
        "threshold": thr, "n": int(len(y)),
        "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
        "accuracy": (tp + tn) / len(y),
        "auroc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def plot_confusion(cm, classes, title, path):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(1.6 * len(classes) + 1.5, 1.6 * len(classes) + 1))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def multiclass_metrics(y, pred, classes) -> dict:
    return {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(len(classes))).tolist(),
    }


def evaluate_all(data_root, out_dir="outputs", cfg: Config | None = None) -> dict:
    """Evaluates every trained model found in out_dir and writes metrics.json plus figures."""
    cfg = cfg or Config()
    out_dir = Path(out_dir)
    _, val_df, test_df = make_splits(build_index(data_root), cfg)
    models = {t: keras.models.load_model(out_dir / f"{t}.keras")
              for t in TASKS if (out_dir / f"{t}.keras").exists()}
    results, thresholds = {}, {}

    for task in ("stage1", "stage2"):
        if task not in models:
            continue
        classes = TASKS[task]["classes"]
        val_f, test_f = task_frame(val_df, task), task_frame(test_df, task)
        thresholds[task] = pick_threshold(val_f["y"], predict(models[task], val_f, task, cfg))
        results[task] = binary_metrics(test_f["y"].to_numpy(), predict(models[task], test_f, task, cfg),
                                       thresholds[task])
        plot_confusion(results[task]["confusion_matrix"], classes, f"{task} (test)",
                       out_dir / f"{task}_confusion.png")

    classes3 = TASKS["three_class"]["classes"]
    test3 = task_frame(test_df, "three_class")
    y3 = test3["y"].to_numpy()

    if "stage1" in models and "stage2" in models:
        # End to end: stage 2 sees every image that stage 1 flags, including stage 1's mistakes.
        p1 = predict(models["stage1"], test3.assign(y=0), "stage1", cfg)
        p2 = predict(models["stage2"], test3.assign(y=0), "stage2", cfg)
        pred = np.where(p1 < thresholds["stage1"], 0, np.where(p2 >= thresholds["stage2"], 2, 1))
        results["two_stage_pipeline"] = multiclass_metrics(y3, pred, classes3)
        plot_confusion(results["two_stage_pipeline"]["confusion_matrix"], classes3,
                       "Two-stage pipeline (test)", out_dir / "two_stage_confusion.png")
        pd.DataFrame({
            "file": [Path(p).name for p in test3["path"]], "true": test3["label"],
            "predicted": [classes3[i] for i in pred],
            "p_pneumonia": p1.round(4), "p_virus_given_pneumonia": p2.round(4), "note": NOTE,
        }).to_csv(out_dir / "test_predictions.csv", index=False)

    if "three_class" in models:
        pred = predict(models["three_class"], test3, "three_class", cfg).argmax(1)
        results["three_class_baseline"] = multiclass_metrics(y3, pred, classes3)
        plot_confusion(results["three_class_baseline"]["confusion_matrix"], classes3,
                       "Single 3-class model (test)", out_dir / "three_class_confusion.png")

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default="outputs")
    a = p.parse_args()
    print(json.dumps(evaluate_all(a.data_root, a.out), indent=2, default=float))
