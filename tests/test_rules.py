import pandas as pd

from src.rules import apply_rules

TEST_CONFIG = {
    "high_amount_threshold": 2000,
    "high_amount_points": 25,
    "night_start_hour": 22,
    "night_end_hour": 6,
    "night_online_points": 20,
    "high_risk_countries": ["NG", "RU", "KP"],
    "high_risk_country_points": 20,
    "risky_categories": ["gift_cards", "crypto", "gambling"],
    "risky_category_points": 15,
    "velocity_count": 4,
    "velocity_points": 20,
    "country_change_points": 15,
}


def test_high_risk_night_online_payment_earns_expected_points_and_reasons():
    df = pd.DataFrame(
        [
            {
                "transaction_id": "risky",
                "timestamp": pd.Timestamp("2024-06-01 23:15:00"),
                "account_id": "acc-r",
                "amount": 5000.0,
                "merchant": "NightMart",
                "merchant_category": "grocery",
                "country": "NG",
                "channel": "online",
            }
        ]
    )
    scored = apply_rules(df, TEST_CONFIG)
    assert int(scored.iloc[0]["rule_points"]) == 65
    reasons = scored.iloc[0]["rule_reasons"]
    assert "High amount (+25)" in reasons
    assert "Night-time online payment (+20)" in reasons
    assert "High-risk country (+20)" in reasons


def test_quiet_in_person_local_payment_fires_no_rules():
    df = pd.DataFrame(
        [
            {
                "transaction_id": "quiet",
                "timestamp": pd.Timestamp("2024-06-01 12:00:00"),
                "account_id": "acc-q",
                "amount": 15.0,
                "merchant": "Corner Shop",
                "merchant_category": "grocery",
                "country": "US",
                "channel": "pos",
            }
        ]
    )
    scored = apply_rules(df, TEST_CONFIG)
    assert int(scored.iloc[0]["rule_points"]) == 0
    assert scored.iloc[0]["rule_reasons"] == ""

