"""If-then checks an analyst can read and tune.

Each fired rule adds points and a short reason. Rules catch known patterns;
they miss brand-new ones, which is why anomaly detection is layered on later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "rules.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _is_night(hour: int, night_start: int, night_end: int) -> bool:
    # Overnight window, e.g. 22:00 until 06:00.
    if night_start >= night_end:
        return hour >= night_start or hour < night_end
    return night_start <= hour < night_end


def apply_rules(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add rule_points and rule_reasons using a frozen or UI-overridden config."""
    if df is None or df.empty:
        empty = df.copy() if df is not None else pd.DataFrame()
        empty["rule_points"] = pd.Series(dtype="int64")
        empty["rule_reasons"] = pd.Series(dtype="object")
        empty["account_velocity"] = pd.Series(dtype="int64")
        return empty

    scored = df.copy()
    high_amount_threshold = float(config["high_amount_threshold"])
    high_amount_points = int(config["high_amount_points"])
    night_start = int(config["night_start_hour"])
    night_end = int(config["night_end_hour"])
    night_online_points = int(config["night_online_points"])
    high_risk_countries = {str(c).upper() for c in config.get("high_risk_countries", [])}
    high_risk_country_points = int(config["high_risk_country_points"])
    risky_categories = {str(c).lower() for c in config.get("risky_categories", [])}
    risky_category_points = int(config["risky_category_points"])
    velocity_count = int(config["velocity_count"])
    velocity_points = int(config["velocity_points"])
    country_change_points = int(config["country_change_points"])

    scored["hour"] = scored["timestamp"].dt.hour
    scored["account_velocity"] = scored.groupby("account_id")["transaction_id"].transform("count")

    first_country_by_account = (
        scored.sort_values("timestamp", kind="mergesort")
        .drop_duplicates(subset=["account_id"], keep="first")
        .set_index("account_id")["country"]
    )
    # Map by account ID so comparison stays aligned even when input row indexes
    # are shuffled or non-consecutive.
    scored["country_changed"] = scored["country"] != scored["account_id"].map(first_country_by_account)

    points: list[int] = []
    reasons: list[str] = []

    for _, row in scored.iterrows():
        row_points = 0
        row_reasons: list[str] = []

        if row["amount"] >= high_amount_threshold:
            row_points += high_amount_points
            row_reasons.append(f"High amount (+{high_amount_points})")

        if row["channel"] == "online" and _is_night(int(row["hour"]), night_start, night_end):
            row_points += night_online_points
            row_reasons.append(f"Night-time online payment (+{night_online_points})")

        if str(row["country"]).upper() in high_risk_countries:
            row_points += high_risk_country_points
            row_reasons.append(f"High-risk country (+{high_risk_country_points})")

        if str(row["merchant_category"]).lower() in risky_categories:
            row_points += risky_category_points
            row_reasons.append(f"Risky merchant category (+{risky_category_points})")

        if int(row["account_velocity"]) >= velocity_count:
            row_points += velocity_points
            row_reasons.append(f"High velocity for account (+{velocity_points})")

        if bool(row["country_changed"]):
            row_points += country_change_points
            row_reasons.append(f"Country change for account (+{country_change_points})")

        points.append(row_points)
        reasons.append("; ".join(row_reasons))

    scored["rule_points"] = points
    scored["rule_reasons"] = reasons
    return scored

