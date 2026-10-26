"""
stage4_evaluate.py — Model evaluation on test set.

Computes comprehensive metrics for a fitted model: accuracy, precision, recall,
F1, specificity, balanced accuracy, MCC, ROC-AUC, PR-AUC, per-day and per-device.
Confusion matrix always uses labels=[0, 1] for correct 2x2 shape.
"""

import time
import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
)

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


def run(model, model_name, X_test, y_test, meta_test=None, run_id=None):
    """
    Evaluate a fitted model on the test set.

    Parameters
    ----------
    model : fitted sklearn estimator
    model_name : str  (e.g. "dt", "rf")
    X_test : ndarray (float32 C-contiguous)
    y_test : ndarray (int8, 0/1)
    meta_test : DataFrame with columns ['mac_src', 'day'], optional
    run_id : str, optional

    Returns
    -------
    dict with keys: 'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix',
        'specificity', 'balanced_accuracy', 'mcc', 'roc_auc', 'pr_auc',
        'class_prevalence', 'per_day', 'per_device'
    """
    t_start = time.perf_counter()
    display = config.MODEL_DISPLAY_NAMES.get(model_name, model_name)

    logger.info("=== STAGE 4: Evaluation [%s] ===", display)
    logger.info("Test set: %s rows", f"{len(X_test):,}")

    # --- Predict in one call (DT/RF are fast, no need for batching) ---
    y_pred = model.predict(X_test)

    # --- Core metrics ---
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1v  = f1_score(y_test, y_pred, zero_division=0)
    # WP2 fix: always produce 2x2 matrix even with single class
    cm   = confusion_matrix(y_test, y_pred, labels=[0, 1])
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Specificity = TN / (TN + FP)
    tn, fp = cm[0, 0], cm[0, 1]
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC-AUC and PR-AUC via predict_proba
    roc_auc = None
    pr_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            if len(np.unique(y_test)) > 1:
                roc_auc = float(roc_auc_score(y_test, y_proba))
                pr_auc = float(average_precision_score(y_test, y_proba))
        except Exception as e:
            logger.warning("Could not compute AUC metrics: %s", e)

    # Class prevalence
    n_total = len(y_test)
    n_iot = int((y_test == 1).sum())
    class_prevalence = {"IoT": n_iot, "non-IoT": n_total - n_iot,
                        "IoT_fraction": float(n_iot / n_total) if n_total > 0 else 0.0}

    logger.info("  Accuracy:  %.4f", acc)
    logger.info("  Precision: %.4f", prec)
    logger.info("  Recall:    %.4f", rec)
    logger.info("  F1:        %.4f", f1v)
    logger.info("  Specificity: %.4f", specificity)
    logger.info("  Balanced Accuracy: %.4f", bal_acc)
    logger.info("  MCC: %.4f", mcc)
    if roc_auc is not None:
        logger.info("  ROC-AUC: %.4f", roc_auc)
    if pr_auc is not None:
        logger.info("  PR-AUC: %.4f", pr_auc)
    logger.info("  Confusion Matrix:\n%s", cm)

    result = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1v),
        "confusion_matrix": cm.tolist(),
        "specificity": specificity,
        "balanced_accuracy": float(bal_acc),
        "mcc": float(mcc),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "class_prevalence": class_prevalence,
    }

    # --- Per-day metrics ---
    per_day = {}
    if meta_test is not None and "day" in meta_test.columns:
        days = meta_test["day"].values
        for day in np.unique(days):
            mask = days == day
            y_t = y_test[mask]
            y_p = y_pred[mask]
            per_day[day] = {
                "n_rows": int(mask.sum()),
                "accuracy": float(accuracy_score(y_t, y_p)),
                "f1": float(f1_score(y_t, y_p, zero_division=0)),
                "precision": float(precision_score(y_t, y_p, zero_division=0)),
                "recall": float(recall_score(y_t, y_p, zero_division=0)),
            }
        logger.info("Per-day metrics computed for %d days", len(per_day))
    result["per_day"] = per_day

    # --- Per-device metrics ---
    per_device = {}
    if meta_test is not None and "mac_src" in meta_test.columns:
        macs = meta_test["mac_src"].values
        for mac in np.unique(macs):
            mask = macs == mac
            y_t = y_test[mask]
            y_p = y_pred[mask]
            true_label = config.MAC_LABEL.get(mac, "unknown")
            correct_frac = float((y_t == y_p).mean())
            per_device[mac] = {
                "true_label": true_label,
                "n_packets": int(mask.sum()),
                "correct_fraction": round(correct_frac, 6),
            }
        logger.info("Per-device metrics computed for %d devices", len(per_device))
    result["per_device"] = per_device

    elapsed = time.perf_counter() - t_start
    logger.info("Stage 4 [%s] completed in %.1f s", display, elapsed)

    return result
