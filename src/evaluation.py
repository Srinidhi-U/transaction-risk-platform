"""Evaluation helpers for datasets that include verified fraud labels."""

from __future__ import annotations

import re

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def normalize_fraud_labels(values: pd.Series) -> pd.Series:
    """Convert common binary label values to 0/1; unknown values become NA."""
    numeric = pd.to_numeric(values, errors="coerce")
    labels = numeric.where(numeric.isin([0, 1]))
    aliases = {
        "true": 1,
        "yes": 1,
        "fraud": 1,
        "fraudulent": 1,
        "false": 0,
        "no": 0,
        "legitimate": 0,
        "genuine": 0,
        "not fraud": 0,
    }
    text_labels = values.astype("string").str.strip().str.lower().map(aliases)
    return labels.fillna(text_labels).astype("Int64")


def compare_score_models(
    labels: pd.Series,
    scores: dict[str, pd.Series],
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Compare candidate scores against verified binary labels at one threshold."""
    rows: list[dict] = []
    matrices: dict[str, dict[str, int]] = {}
    for name, score in scores.items():
        predictions = (score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        rows.append(
            {
                "Approach": name,
                "Precision": precision_score(labels, predictions, zero_division=0),
                "Recall": recall_score(labels, predictions, zero_division=0),
                "F1": f1_score(labels, predictions, zero_division=0),
                "False positives": int(fp),
                "Missed fraud cases": int(fn),
            }
        )
        matrices[name] = {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)}
    return pd.DataFrame(rows), matrices


def false_positive_rate_by_rule(
    scored: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    """Report how often each fired rule matched a non-fraud labeled row."""
    fired: dict[str, list[int]] = {}
    for position, reason_text in enumerate(scored["rule_reasons"].fillna("")):
        for reason in str(reason_text).split("; "):
            rule = re.sub(r" \(\+\d+(?:\.\d+)?\)$", "", reason).strip()
            if rule:
                fired.setdefault(rule, []).append(position)

    rows = []
    legitimate_count = int((labels == 0).sum())
    for rule, positions in fired.items():
        outcomes = labels.iloc[positions]
        rows.append(
            {
                "Rule": rule,
                "Rows triggered": len(positions),
                "False positives": int((outcomes == 0).sum()),
                "False-positive rate": (
                    float((outcomes == 0).sum() / legitimate_count)
                    if legitimate_count
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("False-positive rate", ascending=False) if rows else pd.DataFrame(
        columns=["Rule", "Rows triggered", "False positives", "False-positive rate"]
    )

