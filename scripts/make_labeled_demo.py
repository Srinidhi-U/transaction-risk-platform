"""Generate the reproducible, illustrative synthetic evaluation dataset."""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "sample_transactions_labeled_SYNTHETIC.csv"
SEED = 20261015
ROW_COUNT = 400
FIELDS = [
    "transaction_id",
    "timestamp",
    "account_id",
    "amount",
    "merchant",
    "merchant_category",
    "country",
    "channel",
    "is_fraud",
]


def build_rows() -> list[dict[str, str | int]]:
    """Create 400 labeled rows with 12 fraud examples and intentional overlap."""
    rng = random.Random(SEED)
    merchants = [
        ("FreshMart", "grocery"),
        ("City Fuel", "fuel"),
        ("BookNook", "retail"),
        ("Metro Cafe", "grocery"),
        ("Style Corner", "retail"),
    ]
    rows: list[dict[str, str | int]] = []

    for index in range(ROW_COUNT):
        day = rng.randint(1, 28)
        hour = rng.randint(8, 20)
        minute = rng.choice([0, 15, 30, 45])
        merchant, category = rng.choice(merchants)
        rows.append(
            {
                "transaction_id": f"SYN{index + 1:04d}",
                "timestamp": datetime(2024, 6, day, hour, minute).strftime("%Y-%m-%d %H:%M:%S"),
                "account_id": f"ACC{index + 1:05d}",
                "amount": f"{rng.uniform(8, 950):.2f}",
                "merchant": merchant,
                "merchant_category": category,
                "country": "US",
                "channel": rng.choice(["pos", "online", "atm"]),
                "is_fraud": 0,
            }
        )

    # Four fraud examples deliberately trigger different existing rules.
    fraud_scenarios = [
        (0, "2024-06-02 23:30:00", "ACC90001", "3200.00", "CoinQuick", "crypto", "NG", "online"),
        (1, "2024-06-03 14:00:00", "ACC90002", "2600.00", "Home Electronics", "retail", "US", "pos"),
        (2, "2024-06-05 11:15:00", "ACC90003", "145.00", "Gift Card Hub", "gift_cards", "US", "pos"),
        (3, "2024-06-07 13:45:00", "ACC90004", "85.00", "FreshMart", "grocery", "NG", "online"),
    ]
    for index, timestamp, account, amount, merchant, category, country, channel in fraud_scenarios:
        rows[index].update(
            timestamp=timestamp,
            account_id=account,
            amount=amount,
            merchant=merchant,
            merchant_category=category,
            country=country,
            channel=channel,
            is_fraud=1,
        )

    # Eight subtle fraud cases are ordinary-looking: low amount, US, daytime,
    # POS, common category, and unique account, so none of the configured rules fire.
    for index in range(4, 12):
        rows[index].update(
            timestamp=f"2024-06-{8 + index:02d} {10 + (index % 5):02d}:15:00",
            account_id=f"ACC9{index + 1:04d}",
            amount=f"{rng.uniform(25, 280):.2f}",
            merchant=rng.choice(["FreshMart", "City Fuel", "BookNook"]),
            merchant_category=rng.choice(["grocery", "fuel", "retail"]),
            country="US",
            channel="pos",
            is_fraud=1,
        )

    # Legitimate edge cases intentionally fire rules: four large purchases,
    # one gift-card purchase, one late online purchase, and a travel country change.
    benign_rule_cases = [
        (12, "2024-06-02 12:00:00", "ACC80001", "2450.00", "Home Electronics", "retail", "US", "pos"),
        (13, "2024-06-04 15:00:00", "ACC80002", "3100.00", "Furniture House", "retail", "US", "pos"),
        (14, "2024-06-06 10:00:00", "ACC80003", "2200.00", "Jewelry Corner", "retail", "US", "pos"),
        (15, "2024-06-08 16:00:00", "ACC80004", "2800.00", "Travel Outfitters", "retail", "US", "pos"),
        (16, "2024-06-10 13:00:00", "ACC80005", "75.00", "Gift Card Hub", "gift_cards", "US", "pos"),
        (17, "2024-06-12 23:15:00", "ACC80006", "65.00", "FreshMart Online", "grocery", "US", "online"),
        (18, "2024-06-14 09:00:00", "ACC_TRAVEL_DEMO", "880.00", "Airline Desk", "travel", "US", "online"),
        (19, "2024-06-14 14:00:00", "ACC_TRAVEL_DEMO", "1250.00", "Airline Desk", "travel", "GB", "online"),
    ]
    for index, timestamp, account, amount, merchant, category, country, channel in benign_rule_cases:
        rows[index].update(
            timestamp=timestamp,
            account_id=account,
            amount=amount,
            merchant=merchant,
            merchant_category=category,
            country=country,
            channel=channel,
            is_fraud=0,
        )

    rng.shuffle(rows)
    return rows


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(build_rows())
    print(f"Wrote {ROW_COUNT} synthetic rows to {OUTPUT_PATH}")
    print("Synthetic fraud-label rate: 12 / 400 = 3.0%")


if __name__ == "__main__":
    main()

