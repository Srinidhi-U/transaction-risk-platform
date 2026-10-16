"""Unsupervised unusual-row detector for the uploaded batch.

Isolation Forest does not need labeled fraud. It asks: is this row hard to
group with the others in this file? A high bonus means unusual, not proven
fraud. In production you would train on historical normal traffic instead of
fitting on the same batch you score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

CHANNEL_CODES = {"online": 0, "pos": 1, "atm": 2}


def add_anomaly_bonus(
    df: pd.DataFrame,
    max_bonus: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Add a 0–max_bonus anomaly_bonus column from Isolation Forest scores."""
    scored = df.copy()
    if scored.empty:
        scored["anomaly_bonus"] = pd.Series(dtype="float64")
        return scored

    if len(scored) < 5:
        scored["anomaly_bonus"] = 0.0
        return scored

    features = pd.DataFrame(
        {
            "amount": scored["amount"].astype(float),
            "hour": scored["timestamp"].dt.hour.astype(int),
            "is_weekend": (scored["timestamp"].dt.dayofweek >= 5).astype(int),
            "velocity": scored.get("account_velocity", pd.Series(1, index=scored.index)).astype(float),
            "channel": scored["channel"].map(CHANNEL_CODES).fillna(0).astype(int),
        }
    )
    country_codes, _ = pd.factorize(scored["country"].astype(str))
    features["country"] = country_codes

    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=random_state,
    )
    model.fit(features)
    # decision_function: higher means more typical. Flip so unusual rows score high.
    unusual = -model.decision_function(features)
    low, high = float(np.min(unusual)), float(np.max(unusual))
    if high <= low:
        scaled = np.zeros(len(unusual))
    else:
        scaled = (unusual - low) / (high - low) * max_bonus

    scored["anomaly_bonus"] = np.round(scaled, 1)
    return scored

