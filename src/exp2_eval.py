from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
)


@dataclass(frozen=True)
class EvalSummary:
    n: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    kappa: float


def evaluate(df: pd.DataFrame, gold_col: str, pred_col: str) -> tuple[EvalSummary, dict]:
    gold = df[gold_col].astype(int).to_numpy()
    pred = df[pred_col].astype(int).to_numpy()

    labels = [0, 1, 2, 3, 4]

    acc = float(accuracy_score(gold, pred))
    macro = float(f1_score(gold, pred, average="macro", labels=labels, zero_division=0))
    w = float(f1_score(gold, pred, average="weighted", labels=labels, zero_division=0))
    kappa = float(cohen_kappa_score(gold, pred, labels=labels))

    cm = confusion_matrix(gold, pred, labels=labels)
    rep = classification_report(gold, pred, labels=labels, output_dict=True, zero_division=0)

    return (
        EvalSummary(n=int(len(df)), accuracy=acc, macro_f1=macro, weighted_f1=w, kappa=kappa),
        {"labels": labels, "confusion_matrix": cm.tolist(), "classification_report": rep},
    )


def save_eval(out_path: str | Path, summary: EvalSummary, details: dict) -> None:
    out_path = Path(out_path)

    labels = details["labels"]
    cm = np.array(details["confusion_matrix"])

    lines = []
    lines.append(f"N = {summary.n}")
    lines.append(f"Accuracy      = {summary.accuracy:.4f}")
    lines.append(f"Macro F1      = {summary.macro_f1:.4f}")
    lines.append(f"Weighted F1   = {summary.weighted_f1:.4f}")
    lines.append(f"Cohen's kappa = {summary.kappa:.4f}")
    lines.append("")
    lines.append(f"Labels: {labels}")
    lines.append("")
    lines.append("Confusion Matrix (rows=gold, cols=pred):")
    lines.append(str(cm))
    lines.append("")
    lines.append("Classification report (dict):")
    lines.append(str(details["classification_report"]))

    out_path.write_text("\n".join(lines), encoding="utf-8")