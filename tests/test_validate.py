import pandas as pd

from src.validate import REQUIRED_COLUMNS, validate_and_dedupe


def _row(**overrides):
    base = {
        "transaction_id": "tx-1",
        "timestamp": "2024-06-01 12:00:00",
        "account_id": "acc-1",
        "amount": 25.0,
        "merchant": "Cafe",
        "merchant_category": "grocery",
        "country": "US",
        "channel": "pos",
    }
    base.update(overrides)
    return base


def test_missing_columns_are_reported_and_nothing_is_scored():
    df = pd.DataFrame([{"amount": 10}])
    result = validate_and_dedupe(df)
    assert result.valid.empty
    assert result.missing_columns
    assert "transaction_id" in result.missing_columns
    assert "Missing required column" in str(result.errors.iloc[0]["error"])


def test_non_numeric_and_non_positive_amount_rejected():
    df = pd.DataFrame(
        [
            _row(transaction_id="tx-bad-num", amount="abc"),
            _row(transaction_id="tx-zero", amount=0),
            _row(transaction_id="tx-neg", amount=-5),
            _row(transaction_id="tx-ok", amount=12),
        ]
    )
    result = validate_and_dedupe(df)
    assert list(result.valid["transaction_id"]) == ["tx-ok"]
    errors = " ".join(result.errors["error"].tolist())
    assert "not numeric" in errors
    assert "greater than 0" in errors


def test_bad_timestamp_invalid_channel_and_blank_ids_rejected():
    df = pd.DataFrame(
        [
            _row(transaction_id="tx-ts", timestamp="not-a-date"),
            _row(transaction_id="tx-ch", channel="wire"),
            _row(transaction_id="   ", account_id="acc-1"),
            _row(transaction_id="tx-blank-acc", account_id=""),
            _row(transaction_id="tx-good", timestamp="2024-06-01 13:00:00", channel="online"),
        ]
    )
    result = validate_and_dedupe(df)
    assert list(result.valid["transaction_id"]) == ["tx-good"]
    blob = " ".join(result.errors["error"].tolist())
    assert "timestamp could not be parsed" in blob
    assert "channel must be online, pos, or atm" in blob
    assert "blank transaction_id" in blob
    assert "blank account_id" in blob


def test_invalid_rows_are_excluded_from_valid_set():
    df = pd.DataFrame([_row(amount="x"), _row(transaction_id="keep", amount=9)])
    result = validate_and_dedupe(df)
    assert REQUIRED_COLUMNS[0] in result.valid.columns
    assert "keep" in set(result.valid["transaction_id"])
    assert len(result.valid) == 1


def test_exact_duplicate_transaction_id_keeps_first_and_reports_rest():
    df = pd.DataFrame(
        [
            _row(transaction_id="dup", timestamp="2024-06-01 10:00:00", amount=10),
            _row(transaction_id="dup", timestamp="2024-06-01 11:00:00", amount=99),
            _row(transaction_id="unique", timestamp="2024-06-01 12:00:00", amount=11),
        ]
    )
    result = validate_and_dedupe(df)
    assert list(result.valid["transaction_id"]) == ["dup", "unique"]
    assert result.valid.loc[result.valid["transaction_id"] == "dup", "amount"].iloc[0] == 10
    assert len(result.exact_duplicates) == 1
    assert result.exact_duplicates.iloc[0]["amount"] == 99


def test_near_duplicate_same_account_amount_merchant_within_window():
    df = pd.DataFrame(
        [
            _row(transaction_id="a", timestamp="2024-06-01 10:00:00"),
            _row(transaction_id="b", timestamp="2024-06-01 10:03:00"),
            _row(transaction_id="c", timestamp="2024-06-01 11:00:00"),
        ]
    )
    result = validate_and_dedupe(df, near_duplicate_minutes=5)
    flagged = set(result.near_duplicates["transaction_id"])
    assert flagged == {"b"}
    assert "c" not in flagged
    assert len(result.valid) == 3

