# Real-Time Transaction Risk Detection Platform

**Author:** Uppula Srinidhi

A small assignment app that scores payment rows as soon as you upload a CSV. It is not connected to a live bank feed. “Real-time” here means: validate, score, and explain each valid row in one pass after the file is read.

Scores are **explanations for a reviewer**, not a fraud verdict.

## What you get

1. **CSV upload** (or a bundled sample file)
2. **Input validation** so broken rows never enter scoring
3. **Duplicate handling** for repeated `transaction_id` values and near-duplicates
4. **Configurable risk rules** (YAML + sidebar sliders)
5. **Anomaly detection** with Isolation Forest on the uploaded batch
6. **Explainable risk scores** (0–100) with reason text
7. **Review screen** with analyst actions and CSV export
8. **Optional labeled-data evaluation** with rules-only, anomaly-only, and combined comparisons
9. **Downloadable JSON summary** with risk counts, common rule signals, rejected rows, and available metrics

## Concepts (plain language)

- **Validation vs scoring.** Validation asks “is this row complete and typed correctly?” Scoring asks “does this valid payment look risky?” Invalid rows go to an errors table.
- **Rules** are if-then checks an analyst would write (large amount, night-time online payment, high-risk country, and so on). They are easy to explain. They miss new patterns.
- **Isolation Forest** looks for rows that are statistically unusual compared with the rest of *this file*. It does not need labeled fraud data. Unusual is not the same as fraudulent.
- **Hybrid score** = rule points + a small anomaly bonus, then clipped to 0–100.

Limitation: the model is fit on the uploaded batch, so the file is both the population and the data being scored. A production system would train on historical “normal” traffic instead.

## Transaction fields

| Field | Role |
| --- | --- |
| `transaction_id` | Unique key; exact-duplicate detection |
| `timestamp` | Night/weekend rules and ordering |
| `account_id` | Velocity and country-change checks |
| `amount` | High-value checks |
| `merchant` | Near-duplicate matching |
| `merchant_category` | Risky categories such as `gift_cards`, `crypto`, `gambling` |
| `country` | High-risk country list and country hopping |
| `channel` | Must be `online`, `pos`, or `atm` |

## How scoring works

Default bands (sidebar can change the cutoffs):

- **Low:** below 40
- **Medium:** 40–69
- **High:** 70 and above

Default rules (see `config/rules.yaml`):

- High amount (default threshold 2000)
- Night-time online payment (22:00–06:00)
- High-risk country (`NG`, `RU`, `KP`)
- Risky merchant category
- High velocity (many payments from one account in the file)
- Country change for the same account in the file

Exact duplicate `transaction_id`: keep the row with the earliest timestamp and list the later rows. Near-duplicate (same account, amount, and merchant within 5 minutes): still scored, but flagged.

## Evaluation and false positives

The bundled sample does **not** contain verified fraud outcomes, so the app does not display precision, recall, or false-positive rates for it. If you upload a dataset with an `is_fraud` column, use `1` for confirmed fraud and `0` for confirmed legitimate transactions. The evaluation screen compares:

- **Rules only:** rule points, clipped to 0–100
- **Anomaly only:** the Isolation Forest bonus rescaled to 0–100
- **Combined:** the app's existing hybrid score

The evaluation threshold changes only which scores are counted as predicted fraud for those metrics. It does not change the stored risk score or rules. Missing or unrecognized labels are excluded. For each rule, the false-positive rate is the number of confirmed legitimate rows that rule flagged divided by all confirmed legitimate rows. These results describe the supplied labeled dataset; they do not prove performance on future payments. Use labels whose outcomes are independently verified, and explain where they came from.

Precision means “of the rows flagged, how many were truly fraud?” Recall means “of the known fraud rows, how many did we flag?” A false positive is a legitimate transaction that gets flagged. Lowering the evaluation threshold may find more fraud but can also flag more legitimate payments.

### Synthetic evaluation demo

The sidebar starts with `Labeled demo (synthetic)` selected so reviewers can see the evaluation pipeline on first opening the app. The file `data/sample_transactions_labeled_SYNTHETIC.csv` contains 400 generated examples, including 12 synthetic fraud labels (3%). Some labeled fraud rows look ordinary and do not trigger the configured rules; some labeled legitimate rows intentionally trigger rules, such as a large purchase, a late online payment, or a travel country change. The labels are illustrative inventions, not findings from a bank or verified fraud outcomes. The displayed precision, recall, confusion counts, and false-positive rates are **not real performance estimates** and must not be presented as such.

Recreate the file with the committed generator and its fixed seed:

```bash
python scripts/make_labeled_demo.py
```

Changing a threshold after looking at these same synthetic rows can make the displayed result look better for this file without proving that it will generalize. A meaningful estimate requires separately collected, independently verified labels and evaluation on data held out from threshold or model tuning.

The app also handles common file-reading problems (empty files, malformed CSV text, or text encoding issues) with a readable message. A successful read still needs the required transaction columns shown above. Analyst actions are review notes for the current session and export; they do not change scores or serve as verified fraud labels.

## Project layout

```
app.py                 Streamlit UI
mystartuptv-transparent.png  Transparent MyStartupTV logo
banking-motion.svg     Banking-themed background animation
config/rules.yaml      Default thresholds and point weights
data/sample_transactions.csv
data/sample_transactions_labeled_SYNTHETIC.csv
scripts/make_labeled_demo.py
src/validate.py        Validation and duplicates
src/rules.py           Analyst rules
src/anomalies.py       Isolation Forest bonus
src/evaluation.py      Optional metrics for verified fraud labels
src/scoring.py         Combine into 0–100 + bands
tests/                 pytest helpers only (no UI tests)
```

## Run locally

Python 3.10 or newer.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). Choose a data source in the sidebar: Labeled demo (synthetic), Bundled sample (no labels), or Upload a CSV.

## Tests

Automated checks cover validation, duplicate handling, rule scoring, and risk bands. They call the Python helpers only, not the Streamlit UI.

```bash
pytest
```

## Deployment

Uploads live in the Streamlit **session** only. Nothing is saved on the server.

### Streamlit Community Cloud

1. Put this project on GitHub (initialize git locally if needed, then push).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Create an app:
   - Repository: this project
   - Branch: `main` (or your default branch)
   - Main file: `app.py`
   - Python version: match local (3.11 works)
4. Streamlit Cloud installs `requirements.txt` automatically.

### Temporary public URL from your machine

`streamlit run app.py` is the assignment default. Streamlit can also print a temporary tunnel URL from the app menu. That is optional and not required for marking.

This repo does not include Docker, a custom domain, or a CI pipeline.

## Known limits

- No live payment gateway, login, or database
- No labeled supervised fraud model and no SHAP
- The sample file has no confirmed fraud labels, so it cannot support accuracy claims or meaningful precision/recall values
- Evaluation metrics can be misleading if the supplied labels are incomplete, unrepresentative, or not independently verified
- Isolation Forest results change with the mix of rows in the file
- Analyst actions exist only until the page session ends, unless you download the reviewed CSV

