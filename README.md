# Review‑Risk Predictor (Real Reviews)
Predict **1–2★ review risk** from actual wireless speaker reviews (JBL Flip 6) using logistic regression and random forest.
- Source: `data/reviews.csv` (derived from your uploaded `submittable_reviews (1).csv`).
- Target: `low_star = 1 if rating <= 2 else 0` (class imbalance ~8% low-star).
- Outputs:
  - `reports/metrics.json` — AUC / precision / recall / F1 (per model)
  - `reports/logit_top_coefs.csv` — top tokens increasing low-star risk
  - `reports/rf_feature_importances.csv` — most important features
  - Models saved under `models/`

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python src/train.py
```

## CLI: score new text
```bash
python src/score.py "audio cuts out on subway and battery dies after 2 hours"
```
