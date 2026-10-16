"""Check that uploaded rows look like real transactions before scoring them.

Validation is separate from scoring: a row must be complete and well-typed
first. Invalid rows are listed for the analyst and skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = [
    "transaction_id",
    "timestamp",
    "account_id",
    "amount",
    "merchant",
    "merchant_category",
    "country",
    "channel",
]

ALLOWED_CHANNELS = frozenset({"online", "pos", "atm"})
DEFAULT_NEAR_DUPLICATE_MINUTES = 5


@dataclass
class ValidationResult:
    valid: pd.DataFrame
    errors: pd.DataFrame
    exact_duplicates: pd.DataFrame
    near_duplicates: pd.DataFrame
    missing_columns: list[str] = field(default_factory=list)


def _blank(value) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def validate_and_dedupe(
    df: pd.DataFrame,
    near_duplicate_minutes: int = DEFAULT_NEAR_DUPLICATE_MINUTES,
) -> ValidationResult:
    """Return clean rows, row-level errors, and duplicate reports.

    Exact duplicate `transaction_id` values: keep the first row, set the rest
    aside. Near-duplicates (same account, amount, and merchant within a few
    minutes) stay in the valid set but are flagged for review.
    """
    if df is None or df.empty:
        empty = pd.DataFrame()
        return ValidationResult(empty, empty, empty, empty)

    working = df.copy()
    working.columns = [str(c).strip() for c in working.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in working.columns]
    if missing:
        errors = pd.DataFrame(
            [{"row_number": None, "error": f"Missing required column(s): {', '.join(missing)}"}]
        )
        empty = pd.DataFrame()
        return ValidationResult(empty, errors, empty, empty, missing_columns=missing)

    error_rows: list[dict] = []
    ok_indices: list[int] = []

    for idx, row in working.iterrows():
        problems: list[str] = []

        if _blank(row["transaction_id"]):
            problems.append("blank transaction_id")
        if _blank(row["account_id"]):
            problems.append("blank account_id")
        if _blank(row["merchant"]):
            problems.append("blank merchant")
        if _blank(row["merchant_category"]):
            problems.append("blank merchant_category")
        if _blank(row["country"]):
            problems.append("blank country")

        amount = pd.to_numeric(row["amount"], errors="coerce")
        if pd.isna(amount):
            problems.append("amount is not numeric")
        elif amount <= 0:
            problems.append("amount must be greater than 0")

        timestamp = pd.to_datetime(row["timestamp"], errors="coerce")
        if pd.isna(timestamp):
            problems.append("timestamp could not be parsed")

        channel = "" if _blank(row["channel"]) else str(row["channel"]).strip().lower()
        if channel not in ALLOWED_CHANNELS:
            problems.append("channel must be online, pos, or atm")

        if problems:
            error_rows.append(
                {
                    "row_number": int(idx) + 1 if isinstance(idx, (int, float)) else idx,
                    "transaction_id": row.get("transaction_id"),
                    "error": "; ".join(problems),
                }
            )
        else:
            ok_indices.append(idx)

    errors = pd.DataFrame(error_rows)
    valid = working.loc[ok_indices].copy() if ok_indices else pd.DataFrame(columns=working.columns)

    if valid.empty:
        empty = pd.DataFrame()
        return ValidationResult(valid, errors, empty, empty)

    valid["amount"] = pd.to_numeric(valid["amount"], errors="coerce")
    valid["timestamp"] = pd.to_datetime(valid["timestamp"], errors="coerce")
    valid["channel"] = valid["channel"].astype(str).str.strip().str.lower()
    valid["country"] = valid["country"].astype(str).str.strip().str.upper()
    valid["merchant_category"] = valid["merchant_category"].astype(str).str.strip().str.lower()
    valid["transaction_id"] = valid["transaction_id"].astype(str).str.strip()
    valid["account_id"] = valid["account_id"].astype(str).str.strip()
    valid["merchant"] = valid["merchant"].astype(str).str.strip()

    valid = valid.sort_values("timestamp", kind="mergesort")

    duplicate_mask = valid.duplicated(subset=["transaction_id"], keep="first")
    exact_duplicates = valid.loc[duplicate_mask].copy()
    valid = valid.loc[~duplicate_mask].copy()

    valid["is_near_duplicate"] = False
    if not valid.empty:
        window = pd.Timedelta(minutes=near_duplicate_minutes)
        near_flags = []
        for _, group in valid.groupby(["account_id", "amount", "merchant"], sort=False):
            group = group.sort_values("timestamp")
            previous = None
            for idx, ts in group["timestamp"].items():
                is_near = previous is not None and (ts - previous) <= window
                near_flags.append((idx, is_near))
                previous = ts
        if near_flags:
            flag_index, flag_values = zip(*near_flags)
            valid.loc[list(flag_index), "is_near_duplicate"] = list(flag_values)

    near_duplicates = valid.loc[valid["is_near_duplicate"]].copy()
    valid = valid.reset_index(drop=True)
    exact_duplicates = exact_duplicates.reset_index(drop=True)
    near_duplicates = near_duplicates.reset_index(drop=True)

    return ValidationResult(valid, errors, exact_duplicates, near_duplicates)

