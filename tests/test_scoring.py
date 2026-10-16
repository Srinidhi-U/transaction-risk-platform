from src.scoring import combine_score, risk_band


def test_default_cutoffs_map_low_medium_high():
    assert risk_band(0) == "Low"
    assert risk_band(39.9) == "Low"
    assert risk_band(40) == "Medium"
    assert risk_band(69.9) == "Medium"
    assert risk_band(70) == "High"
    assert risk_band(100) == "High"


def test_combined_score_is_clipped_to_0_100():
    assert combine_score(0, 0) == 0
    assert combine_score(90, 20) == 100
    assert combine_score(-5, 0) == 0
    assert combine_score(40, 5) == 45

