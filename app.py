"""Streamlit UI: upload a payment CSV, score it, and review the risky rows."""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.evaluation import compare_score_models, false_positive_rate_by_rule, normalize_fraud_labels
from src.rules import load_config
from src.scoring import score_transactions
from src.validate import validate_and_dedupe

SAMPLE_PATH = Path(__file__).resolve().parent / "data" / "sample_transactions.csv"
LABELED_DEMO_PATH = Path(__file__).resolve().parent / "data" / "sample_transactions_labeled_SYNTHETIC.csv"
LOGO_PATH = Path(__file__).resolve().parent / "mystartuptv-transparent.png"
MOTION_BACKGROUND_PATH = Path(__file__).resolve().parent / "banking-motion.svg"
DISPLAY_COLUMNS = [
    "transaction_id",
    "timestamp",
    "account_id",
    "amount",
    "merchant",
    "merchant_category",
    "country",
    "channel",
    "risk_score",
    "risk_band",
    "reasons",
]


def _config_from_sidebar(base: dict) -> dict:
    config = deepcopy(base)
    st.sidebar.header("Risk rules")
    st.sidebar.caption("These are the same knobs as config/rules.yaml. Change them to see scores move.")

    config["high_amount_threshold"] = st.sidebar.number_input(
        "High amount threshold", min_value=1.0, value=float(base["high_amount_threshold"]), step=50.0
    )
    config["high_amount_points"] = st.sidebar.slider("High amount points", 0, 40, int(base["high_amount_points"]))
    config["night_online_points"] = st.sidebar.slider(
        "Night-time online points", 0, 40, int(base["night_online_points"])
    )
    config["high_risk_country_points"] = st.sidebar.slider(
        "High-risk country points", 0, 40, int(base["high_risk_country_points"])
    )
    config["risky_category_points"] = st.sidebar.slider(
        "Risky category points", 0, 40, int(base["risky_category_points"])
    )
    config["velocity_count"] = st.sidebar.number_input(
        "Velocity: payments per account to flag", min_value=2, value=int(base["velocity_count"]), step=1
    )
    config["velocity_points"] = st.sidebar.slider("Velocity points", 0, 40, int(base["velocity_points"]))
    config["country_change_points"] = st.sidebar.slider(
        "Country-change points", 0, 40, int(base["country_change_points"])
    )
    config["medium_cutoff"] = st.sidebar.slider("Medium band starts at", 1, 90, int(base["medium_cutoff"]))
    config["high_cutoff"] = st.sidebar.slider("High band starts at", 10, 100, int(base["high_cutoff"]))
    config["anomaly_max_bonus"] = st.sidebar.slider(
        "Max anomaly bonus", 0, 40, int(base["anomaly_max_bonus"])
    )
    config["near_duplicate_minutes"] = int(base.get("near_duplicate_minutes", 5))
    return config


class CSVLoadError(ValueError):
    """A user-readable CSV loading problem."""


def _load_frame(uploaded, data_source: str) -> pd.DataFrame | None:
    if data_source == "Bundled sample (no labels)":
        source = SAMPLE_PATH
    elif data_source == "Labeled demo (synthetic)":
        source = LABELED_DEMO_PATH
    elif data_source == "Upload a CSV":
        if uploaded is None:
            return None
        source = uploaded
    else:
        return None

    try:
        if hasattr(source, "seek"):
            source.seek(0)
        try:
            frame = pd.read_csv(source, encoding="utf-8-sig")
        except UnicodeDecodeError:
            if hasattr(source, "seek"):
                source.seek(0)
            frame = pd.read_csv(source, encoding="cp1252")
    except pd.errors.EmptyDataError as exc:
        raise CSVLoadError("The file is empty. Add a header row and transaction data, then upload it again.") from exc
    except pd.errors.ParserError as exc:
        raise CSVLoadError("The file could not be read as a table. Check that it is a valid CSV file.") from exc
    except (OSError, UnicodeError, ValueError) as exc:
        raise CSVLoadError("The file could not be opened. Save it as a CSV file and try again.") from exc

    if frame.empty and len(frame.columns) == 0:
        raise CSVLoadError("The file does not contain a readable CSV header or transaction data.")
    return frame


def _summary_report(raw, result, scored, evaluation=None) -> bytes:
    """Build a portable JSON summary for this upload."""
    bands = scored["risk_band"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    rule_counts: dict[str, int] = {}
    for reasons in scored.get("rule_reasons", pd.Series(dtype=str)).fillna(""):
        for reason in str(reasons).split("; "):
            if reason:
                name = reason.rsplit(" (+", 1)[0]
                rule_counts[name] = rule_counts.get(name, 0) + 1
    report = {
        "rows_in_file": int(len(raw)),
        "valid_scored_rows": int(len(result.valid)),
        "invalid_rows_rejected": int(len(result.errors)),
        "exact_duplicate_rows_dropped": int(len(result.exact_duplicates)),
        "near_duplicate_rows_flagged": int(len(result.near_duplicates)),
        "risk_band_counts": {str(key): int(value) for key, value in bands.items()},
        "most_triggered_rules": [
            {"rule": name, "transactions": count}
            for name, count in sorted(rule_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        ],
        "evaluation": evaluation,
        "evaluation_note": "Metrics are included only when verified is_fraud labels were provided.",
    }
    return json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")


def _metric_card(label: str, value: int, tone: str) -> None:
    """Render a compact, color-coded metric without changing its value."""
    st.markdown(
        f'<div class="metric-card metric-{tone}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value:,}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Transaction Risk Detection", layout="wide")
logo_data = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii") if LOGO_PATH.exists() else ""
motion_background_data = (
    base64.b64encode(MOTION_BACKGROUND_PATH.read_bytes()).decode("ascii")
    if MOTION_BACKGROUND_PATH.exists()
    else ""
)
st.markdown(
    """
    <style>
    :root {
        --surface: #111827;
        --surface-raised: #172235;
        --line: rgba(148, 163, 184, 0.18);
        --muted: #a8b5c8;
        --blue: #f4b41a;
        --teal: #ffd166;
    }
    .stApp {
        background-color: #11100e;
        background-image:
            radial-gradient(ellipse at 12% 12%, rgba(255, 190, 46, 0.27), transparent 36%),
            radial-gradient(ellipse at 88% 26%, rgba(249, 115, 22, 0.21), transparent 34%),
            radial-gradient(ellipse at 66% 92%, rgba(255, 209, 102, 0.18), transparent 40%),
            linear-gradient(135deg, #11100e 0%, #21180b 48%, #11100e 100%),
            url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNDAwIDcwMCI+PGcgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZjRiNDFhIiBzdHJva2Utb3BhY2l0eT0iMC4wOCI+PGNpcmNsZSBjeD0iNzAwIiBjeT0iMzUwIiByPSIyNjUiIHN0cm9rZS13aWR0aD0iMiIvPjxjaXJjbGUgY3g9IjcwMCIgY3k9IjM1MCIgcj0iMjkwIiBzdHJva2Utd2lkdGg9IjEiIHN0cm9rZS1kYXNoYXJyYXk9IjQgMTgiLz48L2c+PHRleHQgeD0iNzAwIiB5PSIzODUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIHRyYW5zZm9ybT0icm90YXRlKC0xOCA3MDAgMzUwKSIgZm9udC1mYW1pbHk9IkFyaWFsLHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEyIiBmb250LXdlaWdodD0iNzAwIiBsZXR0ZXItc3BhY2luZz0iMTQiIGZpbGw9IiNmNGI0MWEiIGZpbGwtb3BhY2l0eT0iMC4wNDUiPk1ZIFNUQVJUVVAgVFY8L3RleHQ+PC9zdmc+"),
            url("data:image/svg+xml;base64,<!--MOTION_BACKGROUND-->");
        background-size: 210% 210%, 190% 190%, 200% 200%, auto, 100% 100%, 135% 135%;
        background-attachment: fixed;
        animation: ambient-drift 14s ease-in-out infinite alternate;
    }
    @keyframes ambient-drift {
        from { background-position: 0% 0%, 100% 0%, 45% 100%, 0 0, center center, 0% 0%; }
        to { background-position: 100% 72%, 0% 100%, 100% 0%, 0 0, center center, 100% 100%; }
    }
    .stApp, .stApp button, .stApp input, .stApp textarea {
        font-family: "Aptos", "Segoe UI", system-ui, sans-serif;
    }
    h1, h2, h3, .hero-title {
        font-family: "Aptos Display", "Segoe UI Variable Display", "Segoe UI", sans-serif;
        letter-spacing: -0.025em;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1711 0%, #11100e 100%);
        border-right: 1px solid var(--line);
    }
    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .brand-strip {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        min-height: 52px;
        margin: 1.5rem 0 0.75rem;
    }
    .brand-strip img {
        display: block;
        width: 120px;
        max-height: 52px;
        object-fit: contain;
        filter: drop-shadow(0 0 8px rgba(244, 180, 26, 0.22));
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.2rem;
        margin: 0 0 1.5rem;
        border: 1px solid rgba(244, 180, 26, 0.3);
        border-radius: 20px;
        background:
            radial-gradient(circle at 92% 18%, rgba(255, 209, 102, 0.14), transparent 24%),
            linear-gradient(120deg, rgba(48, 38, 19, 0.98), rgba(25, 22, 16, 0.92));
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.2);
    }
    .hero > * { position: relative; z-index: 1; }
    .hero::before {
        content: "";
        position: absolute;
        width: 150px;
        height: 150px;
        right: 20%;
        bottom: -92px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(244, 180, 26, 0.3), rgba(244, 180, 26, 0));
        pointer-events: none;
        animation: float-glow 10s ease-in-out infinite alternate;
    }
    @keyframes float-glow {
        from { transform: translate3d(-10px, 0, 0); opacity: 0.45; }
        to { transform: translate3d(24px, -20px, 0); opacity: 0.9; }
    }
    .hero::after {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        right: -65px;
        top: -105px;
        border: 1px solid rgba(255, 209, 102, 0.24);
        border-radius: 50%;
        box-shadow: 0 0 0 24px rgba(255, 209, 102, 0.035), 0 0 0 48px rgba(255, 209, 102, 0.025);
        pointer-events: none;
        animation: halo-drift 22s ease-in-out infinite alternate;
    }
    @keyframes halo-drift {
        from { transform: translate3d(0, 0, 0) scale(0.96); opacity: 0.72; }
        to { transform: translate3d(-18px, 14px, 0) scale(1.04); opacity: 1; }
    }
    .hero-kicker {
        margin: 0 0 0.55rem;
        color: var(--teal);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        margin: 0 0.45rem 1px 0;
        border-radius: 50%;
        background: var(--teal);
        box-shadow: 0 0 0 0 rgba(255, 209, 102, 0.55);
        animation: beacon-pulse 2.4s ease-out infinite;
    }
    @keyframes beacon-pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 209, 102, 0.45); }
        70%, 100% { box-shadow: 0 0 0 9px rgba(255, 209, 102, 0); }
    }
    .hero-title {
        margin: 0;
        color: #f8fafc;
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 750;
        letter-spacing: -0.04em;
        line-height: 1.08;
    }
    .hero-title span { color: var(--blue); }
    .hero-copy {
        max-width: 760px;
        margin: 0.8rem 0 0;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.65;
    }
    .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1.1rem;
    }
    .hero-tags span {
        padding: 0.35rem 0.7rem;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.42);
        color: #d5e2f2;
        font-size: 0.78rem;
    }
    .hero { display: flex; align-items: center; justify-content: space-between; gap: 2rem; }
    .hero-copyblock { flex: 1; min-width: 0; }
    .metric-card {
        min-height: 94px;
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: linear-gradient(145deg, rgba(23, 34, 53, 0.96), rgba(17, 24, 39, 0.92));
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-left: 4px solid var(--card-accent);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.14);
        transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: var(--card-accent);
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.2);
    }
    .metric-blue {
        --card-accent: #60a5fa;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.27), rgba(17, 24, 39, 0.95) 76%);
        border-color: rgba(96, 165, 250, 0.28);
    }
    .metric-green {
        --card-accent: #34d399;
        background: linear-gradient(135deg, rgba(5, 150, 105, 0.24), rgba(17, 24, 39, 0.95) 76%);
        border-color: rgba(52, 211, 153, 0.26);
    }
    .metric-red {
        --card-accent: #fb7185;
        background: linear-gradient(135deg, rgba(225, 29, 72, 0.24), rgba(17, 24, 39, 0.95) 76%);
        border-color: rgba(251, 113, 133, 0.28);
    }
    .metric-amber {
        --card-accent: #fbbf24;
        background: linear-gradient(135deg, rgba(217, 119, 6, 0.25), rgba(17, 24, 39, 0.95) 76%);
        border-color: rgba(251, 191, 36, 0.28);
    }
    .metric-label { color: var(--muted); font-size: 0.88rem; font-weight: 600; }
    .metric-value { margin-top: 0.3rem; color: #f8fafc; font-size: 1.85rem; font-weight: 750; line-height: 1.1; }
    div[role="tablist"] { gap: 1.25rem; border-bottom-color: var(--line); }
    button[data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 0.8rem 1.05rem;
        font-size: 1rem;
        font-weight: 650;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--blue);
        border-bottom-color: var(--blue);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stFileUploader"] section {
        border: 1px dashed rgba(96, 165, 250, 0.45);
        border-radius: 14px;
        background: rgba(23, 34, 53, 0.55);
    }
    .stDownloadButton button {
        border: 1px solid rgba(96, 165, 250, 0.4);
        border-radius: 10px;
    }
    @media (max-width: 700px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .block-container { padding-top: 1rem; }
        .brand-strip { min-height: 44px; margin-top: 1.25rem; }
        .brand-strip img { width: 100px; max-height: 44px; }
        .hero { padding: 1.4rem; border-radius: 16px; margin-top: 0; }
    }
    @media (prefers-reduced-motion: reduce) {
        .stApp, .hero::before, .hero::after, .pulse-dot { animation: none !important; }
        .metric-card { transition: none; }
    }
    </style>
    <div class="brand-strip"><!--BRAND_LOGO--></div>
    <div class="hero">
      <div class="hero-copyblock">
      <p class="hero-kicker"><span class="pulse-dot"></span>Payment risk analysis · Analyst workspace</p>
      <h1 class="hero-title">Transaction Risk <span>Platform</span></h1>
      <p class="hero-copy">Upload a payment CSV or explore the sample. Review explainable risk signals and data quality in one place. Scores highlight unusual activity; they are not a fraud verdict.</p>
      <div class="hero-tags"><span>CSV screening</span><span>Explainable rules</span><span>Analyst review</span></div>
      </div>
    </div>
    """.replace("<!--BRAND_LOGO-->", f'<img src="data:image/png;base64,{logo_data}" alt="MyStartupTV logo">' if logo_data else "").replace("<!--MOTION_BACKGROUND-->", motion_background_data),
    unsafe_allow_html=True,
)

base_config = load_config()
config = _config_from_sidebar(base_config)

st.sidebar.markdown("---")
data_source = st.sidebar.radio(
    "Choose transaction data",
    ["Labeled demo (synthetic)", "Bundled sample (no labels)", "Upload a CSV"],
    index=0,
)
uploaded = (
    st.sidebar.file_uploader("Upload a CSV", type=["csv"])
    if data_source == "Upload a CSV"
    else None
)
try:
    raw = _load_frame(uploaded, data_source=data_source)
except CSVLoadError as exc:
    st.error(str(exc))
    st.info("Upload a readable .csv file with a header row and transaction records.")
    st.stop()

if raw is None:
    st.info("Choose one of the sample files in the sidebar, or select Upload a CSV and choose a file.")
    st.stop()

if data_source == "Labeled demo (synthetic)":
    st.warning(
        "Synthetic demo data — these labels are invented to demonstrate the evaluation workflow. "
        "The metrics are not real fraud-performance estimates.",
        icon="⚠️",
    )

result = validate_and_dedupe(raw, near_duplicate_minutes=int(config["near_duplicate_minutes"]))

if result.missing_columns:
    st.error("This file is missing required columns: " + ", ".join(result.missing_columns))
    st.stop()

if result.valid.empty:
    st.warning("No valid rows to score. Fix the errors below and try again.")
else:
    scored = score_transactions(result.valid, config)
    if "review_actions" not in st.session_state:
        st.session_state.review_actions = {}

upload_tab, dashboard_tab, review_tab = st.tabs(["Upload & quality", "Score & dashboard", "Review"])

with upload_tab:
    st.subheader("File quality")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric_card("Rows in file", len(raw), "blue")
    with c2:
        _metric_card("Valid (scored)", len(result.valid), "green")
    with c3:
        _metric_card("Invalid rows", len(result.errors), "red")
    with c4:
        _metric_card("Exact duplicate IDs dropped", len(result.exact_duplicates), "amber")

    st.caption(
        "Invalid rows never enter scoring. Exact duplicate transaction IDs keep the earliest timestamp. "
        "Near-duplicates (same account, amount, and merchant within a few minutes) are still scored but flagged."
    )

    if not result.errors.empty:
        st.markdown("**Validation errors**")
        st.dataframe(result.errors, width="stretch", hide_index=True)
    if not result.exact_duplicates.empty:
        st.markdown("**Dropped exact duplicates**")
        st.dataframe(
            result.exact_duplicates[["transaction_id", "timestamp", "account_id", "amount", "merchant"]],
            width="stretch",
            hide_index=True,
        )
    if not result.near_duplicates.empty:
        st.markdown("**Near-duplicates still in the scored set**")
        st.dataframe(
            result.near_duplicates[["transaction_id", "timestamp", "account_id", "amount", "merchant"]],
            width="stretch",
            hide_index=True,
        )

with dashboard_tab:
    if result.valid.empty:
        st.info("Nothing to chart until validation produces valid rows.")
    else:
        bands = scored["risk_band"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int)
        b1, b2, b3 = st.columns(3)
        with b1:
            _metric_card("High risk", int(bands.get("High", 0)), "red")
        with b2:
            _metric_card("Medium risk", int(bands.get("Medium", 0)), "amber")
        with b3:
            _metric_card("Low risk", int(bands.get("Low", 0)), "green")

        st.markdown("**Risk score distribution**")
        buckets = pd.cut(
            scored["risk_score"],
            bins=[-0.1, 20, 40, 60, 80, 100],
            labels=["0-20", "20-40", "40-60", "60-80", "80-100"],
        )
        st.bar_chart(buckets.value_counts().sort_index(), height=220)

        left, right = st.columns(2)
        with left:
            st.markdown("**Highest-risk accounts**")
            by_account = (
                scored.groupby("account_id", as_index=False)
                .agg(avg_score=("risk_score", "mean"), payments=("transaction_id", "count"))
                .sort_values("avg_score", ascending=False)
                .head(10)
            )
            st.dataframe(by_account, width="stretch", hide_index=True)
        with right:
            st.markdown("**Highest-risk merchants**")
            by_merchant = (
                scored.groupby("merchant", as_index=False)
                .agg(avg_score=("risk_score", "mean"), payments=("transaction_id", "count"))
                .sort_values("avg_score", ascending=False)
                .head(10)
            )
            st.dataframe(by_merchant, width="stretch", hide_index=True)

        st.markdown("**All scored transactions**")
        visible_columns = DISPLAY_COLUMNS + (["is_fraud"] if "is_fraud" in scored.columns else [])
        st.dataframe(
            scored[visible_columns],
            width="stretch",
            hide_index=True,
            column_config={
                "amount": st.column_config.NumberColumn(
                    "Amount (₹)", format="₹ %.2f"
                )
            },
        )

        st.markdown("### Optional model evaluation")
        st.caption(
            "This section works only when the uploaded CSV includes a verified `is_fraud` column "
            "(1 = confirmed fraud, 0 = confirmed legitimate). The bundled sample has no verified labels."
        )
        if "is_fraud" not in scored.columns:
            st.info("No verified labels found, so precision and recall cannot be calculated for this file.")
            evaluation_for_report = None
        else:
            labels_all = normalize_fraud_labels(scored["is_fraud"])
            labels = labels_all.dropna().astype(int)
            if labels.empty:
                st.warning("The is_fraud column has no recognized 0/1 or fraud/legitimate labels.")
                evaluation_for_report = None
            else:
                threshold = st.slider(
                    "Evaluation decision threshold (score only)",
                    min_value=0,
                    max_value=100,
                    value=50,
                    help="Changes the evaluation prediction cutoff only; it does not change risk scores or rules.",
                )
                if len(labels) < len(scored):
                    st.caption(f"Evaluating {len(labels)} labeled rows; {len(scored) - len(labels)} unlabeled rows are excluded.")
                st.caption(
                    f"Label counts: {int((labels == 1).sum())} confirmed fraud and "
                    f"{int((labels == 0).sum())} confirmed legitimate."
                )

                labeled_scored = scored.loc[labels.index]
                max_bonus = float(config.get("anomaly_max_bonus", 0))
                ml_only = (
                    labeled_scored["anomaly_bonus"] / max_bonus * 100
                    if max_bonus > 0
                    else pd.Series(0.0, index=labeled_scored.index)
                )
                score_comparison, confusion = compare_score_models(
                    labels,
                    {
                        "Rules only": labeled_scored["rule_points"].clip(0, 100),
                        "ML anomaly only": ml_only,
                        "Combined (hybrid)": labeled_scored["risk_score"],
                    },
                    threshold=float(threshold),
                )
                st.caption(
                    "Precision answers: of the rows flagged, how many were fraud? "
                    "Recall answers: of the known fraud rows, how many were flagged?"
                )
                st.dataframe(
                    score_comparison.style.format(
                        {"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}"}
                    ),
                    width="stretch",
                    hide_index=True,
                )
                st.markdown("**Confusion matrix counts**")
                st.dataframe(pd.DataFrame(confusion).T, width="stretch")

                rule_rates = false_positive_rate_by_rule(labeled_scored.reset_index(drop=True), labels.reset_index(drop=True))
                if rule_rates.empty:
                    st.info("No rule-based signals fired among the labeled rows.")
                else:
                    st.markdown("**False positives for each fired rule**")
                    st.caption("Rate = legitimate labeled transactions flagged by that rule ÷ all legitimate labeled transactions.")
                    st.dataframe(
                        rule_rates.style.format({"False-positive rate": "{:.1%}"}),
                        width="stretch",
                        hide_index=True,
                    )
                evaluation_for_report = {
                    "labeled_rows": int(len(labels)),
                    "decision_threshold": int(threshold),
                    "approaches": score_comparison.assign(
                        Precision=score_comparison["Precision"].round(4),
                        Recall=score_comparison["Recall"].round(4),
                        F1=score_comparison["F1"].round(4),
                    ).to_dict(orient="records"),
                    "confusion_counts": confusion,
                }

        st.download_button(
            "Download summary report (JSON)",
            data=_summary_report(raw, result, scored, evaluation_for_report),
            file_name="transaction_risk_summary.json",
            mime="application/json",
        )
with review_tab:
    if result.valid.empty:
        st.info("Nothing to review yet.")
    else:
        bands_filter = st.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium"])
        account_query = st.text_input("Search account_id")
        view = scored[scored["risk_band"].isin(bands_filter)] if bands_filter else scored
        if account_query.strip():
            view = view[view["account_id"].str.contains(account_query.strip(), case=False, na=False)]

        if view.empty:
            st.warning("No rows match those filters.")
        else:
            ids = view["transaction_id"].tolist()
            selected_id = st.selectbox("Select a transaction", ids)
            selected = view.loc[view["transaction_id"] == selected_id].iloc[0]

            st.markdown(f"**{selected_id}** — {selected['risk_band']} ({selected['risk_score']})")
            st.write(selected["reasons"] or "No rules fired; score comes only from the batch anomaly bonus if any.")
            detail = selected[DISPLAY_COLUMNS].to_frame("value")
            st.dataframe(detail, width="stretch")

            action = st.radio(
                "Analyst action",
                ["pending", "clear", "escalate", "confirm_risk"],
                index=["pending", "clear", "escalate", "confirm_risk"].index(
                    st.session_state.review_actions.get(selected_id, "pending")
                ),
                horizontal=True,
            )
            st.caption("Analyst action is a review note. It does not change the risk score or prove a transaction is fraud.")
            st.session_state.review_actions[selected_id] = action

            reviewed = scored[scored["risk_band"].isin(["High", "Medium"])].copy()
            reviewed["analyst_action"] = reviewed["transaction_id"].map(
                lambda txn_id: st.session_state.review_actions.get(txn_id, "pending")
            )
            st.markdown("**Queue**")
            st.dataframe(reviewed[DISPLAY_COLUMNS + ["analyst_action"]], width="stretch", hide_index=True)
            csv_bytes = reviewed.to_csv(index=False).encode("utf-8")
            st.download_button("Download reviewed CSV", csv_bytes, "reviewed_transactions.csv", "text/csv")

