# ECMC Pipeline Risk Model

A machine-learning project, funded by the Colorado Energy and Carbon Management Commission
(ECMC), to rank the failure risk of oil & gas flowlines/pipelines across Colorado.

The model is trained on a pipeline-level dataset combining ECMC flowline registrations with
confirmed spill/failure records. It classifies every pipeline into one of four risk classes —
**Low, Medium, High, Severe** — based on a continuous, calibrated failure-probability score.

## Contents

- [`ECMC_Pipeline_Risk_Model.ipynb`](ECMC_Pipeline_Risk_Model.ipynb) — end-to-end, already-executed
  notebook: data loading, exploratory KDE plots, binary training (5 imbalance-aware classifiers),
  leak-free cross-validation, probability calibration, derivation of the Low/Medium/High/Severe
  thresholds, and an interactive tool at the end to score a new pipeline's specifications.
- `Data/` — input data:
  - `pipeline_level_dataset.csv` — one row per pipeline (age, diameter, MAOP, elevation, length,
    material, fluid, status, location type, confirmed failure count)
  - `Flowline-Related Spills (through 2024).xlsx` — ECMC spill records
  - `FlowlineSpreadsheet_Mines.xlsx` — ECMC flowline registration records
- `requirements.txt` — exact package versions used to build and run the notebook

## Method summary

1. **Phase 1 — Binary training**: label = 1 if a pipeline has ≥1 confirmed failure, else 0.
   Five imbalance-aware models (Logistic Regression, Random Forest, SVM, XGBoost, Stacking
   Ensemble) are trained on all 11 pipeline features with BorderlineSMOTE + TomekLinks
   resampling; SMOTE is applied inside each cross-validation fold to avoid leakage.
2. **Phase 2 — Continuous risk score**: the best model by PR-AUC (XGBoost) is probability-
   calibrated with isotonic regression, producing a `risk_score` in `[0, 1]` for every pipeline.
3. **Phase 3 — Four-class assignment**: thresholds `T1 < T2 < T3` derived from the score
   distribution split every pipeline into **Low / Medium / High / Severe**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**macOS only** — XGBoost needs the OpenMP runtime:

```bash
brew install libomp
```

Then launch the notebook:

```bash
jupyter notebook ECMC_Pipeline_Risk_Model.ipynb
```

## Assessing a new pipeline

The last section of the notebook loads the trained preprocessor, calibrated model, and
thresholds, then exposes an `assess_pipeline(spec)` function plus an interactive
ipywidgets form. Give it a pipeline's specs (age, MAOP, diameter, elevation, length,
material, fluid, status, etc.) and it returns a risk score and a Low / Medium / High /
Severe classification.
