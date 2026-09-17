"""
Extra evaluation metrics computed once at the end of training: confusion
matrix, precision/recall/F1 (per-class + macro/weighted) and ROC-AUC.

Kept in its own file so it doesn't touch the existing training loop logic
in trainer.py -- it's just called once at the end.

Printed twice: once as a readable table, and once as a plain Python dict
literal you can copy straight into another script/notebook to make your
own plots.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


@torch.no_grad()
def compute_metrics(model, loader, device, class_names: list[str] | None = None) -> dict:
    """Run `model` over `loader` once and return a dict of classification metrics."""
    model.eval()
    all_logits, all_true = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        all_logits.append(logits.cpu())
        all_true.append(yb.cpu())

    logits = torch.cat(all_logits)
    y_true = torch.cat(all_true).numpy()
    probs = F.softmax(logits, dim=-1).numpy()
    y_pred = probs.argmax(-1)

    n_classes = probs.shape[1]
    labels = list(range(n_classes))
    if class_names is None:
        class_names = [str(i) for i in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    accuracy = float((y_true == y_pred).mean())

    try:
        if n_classes == 2:
            roc_auc = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            roc_auc = float(roc_auc_score(y_true, probs, multi_class="ovr"))
    except ValueError:
        # only one class present in y_true for this split -- AUC undefined
        roc_auc = float("nan")

    return {
        "accuracy": accuracy,
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "roc_auc": roc_auc,
        "per_class": {
            class_names[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(n_classes)
        },
        "confusion_matrix": cm.tolist(),
    }


def print_metrics(metrics: dict, title: str = "FINAL TEST METRICS") -> None:
    """Readable table + a copy-paste-ready `RESULTS = {...}` python dict."""
    print("\n" + "=" * 58)
    print(title)
    print("=" * 58)
    print(f"accuracy         = {metrics['accuracy']:.4f}")
    print(f"f1_macro         = {metrics['f1_macro']:.4f}")
    print(f"f1_weighted      = {metrics['f1_weighted']:.4f}")
    print(f"precision_macro  = {metrics['precision_macro']:.4f}")
    print(f"recall_macro     = {metrics['recall_macro']:.4f}")
    print(f"roc_auc          = {metrics['roc_auc']:.4f}")
    print("-" * 58)
    print("per-class:")
    for name, v in metrics["per_class"].items():
        print(f"  class {name}: precision={v['precision']:.4f}  "
              f"recall={v['recall']:.4f}  f1={v['f1']:.4f}  support={v['support']}")
    print("-" * 58)
    print("confusion_matrix (rows=true, cols=pred):")
    for row in metrics["confusion_matrix"]:
        print(" ", row)
    print("=" * 58)

    print("\n# copy-paste block:")
    print("RESULTS = {")
    print(f"    \"accuracy\": {metrics['accuracy']:.4f},")
    print(f"    \"f1_macro\": {metrics['f1_macro']:.4f},")
    print(f"    \"f1_weighted\": {metrics['f1_weighted']:.4f},")
    print(f"    \"precision_macro\": {metrics['precision_macro']:.4f},")
    print(f"    \"recall_macro\": {metrics['recall_macro']:.4f},")
    print(f"    \"roc_auc\": {metrics['roc_auc']:.4f},")
    print(f"    \"confusion_matrix\": {metrics['confusion_matrix']},")
    print("    \"per_class\": {")
    for name, v in metrics["per_class"].items():
        print(f"        \"{name}\": {{\"precision\": {v['precision']:.4f}, "
              f"\"recall\": {v['recall']:.4f}, \"f1\": {v['f1']:.4f}, "
              f"\"support\": {v['support']}}},")
    print("    },")
    print("}")
