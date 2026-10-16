"""Blend rule points with a small anomaly bonus into a 0–100 score.

The number is an explanation aid for reviewers, not a courtroom verdict.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.anomalies import add_anomaly_bonus
from src.rules import apply_rules


def risk_band(score: float, medium_cutoff: int = 40, high_cutoff: int = 70) -> str:
    if score >= high_cutoff:
        return "High"
    if score >= medium_cutoff:
        return "Medium"
    return "Low"


def combine_score(rule_points: float, anomaly_bonus: float) -> float:
    """Clip the hybrid score so the UI always shows 0–100."""
    return float(max(0, min(100, rule_points + anomaly_bonus)))


def score_transactions(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    scored = apply_rules(df, config)
    scored = add_anomaly_bonus(scored, max_bonus=int(config.get("anomaly_max_bonus", 30)))
    medium = int(config.get("medium_cutoff", 40))
    high = int(config.get("high_cutoff", 70))

    scores = []
    bands = []
    reasons = []
    for _, row in scored.iterrows():
        bonus = float(row.get("anomaly_bonus", 0) or 0)
        total = combine_score(float(row.get("rule_points", 0) or 0), bonus)
        scores.append(round(total, 1))
        bands.append(risk_band(total, medium, high))

        parts = []
        if row.get("rule_reasons"):
            parts.append(str(row["rule_reasons"]))
        if bonus > 0:
            parts.append(f"Unusual compared with this batch (+{bonus:.1f})")
        if row.get("is_near_duplicate"):
            parts.append("Near-duplicate of a recent similar payment")
        reasons.append("; ".join(p for p in parts if p))

    scored["risk_score"] = scores
    scored["risk_band"] = bands
    scored["reasons"] = reasons
    return scored

