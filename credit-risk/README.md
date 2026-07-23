# Credit Risk — PD Modelling and an IFRS 9 Expected Credit Loss Layer

End-to-end credit risk modelling on the most-used Kaggle credit risk dataset
(32,581 consumer loans), carried all the way to the number a bank actually books:
a staged, scenario-weighted **IFRS 9 expected credit loss** with every assumption
stated and stress-tested.

## Why this framing

Most credit-risk portfolio projects stop at a classifier and an AUC. But a probability
of default is an *input*, not a deliverable — it feeds provisioning, and provisioning
has its own machinery: calibration (a PD must be an honest probability), staging
(12-month vs lifetime loss), LGD/EAD assumptions, discounting at the effective interest
rate, and probability-weighted forward-looking scenarios. This repository builds the
whole chain:

```
EDA + statistical association battery  →  tuned & calibrated XGBoost PD
        →  SICR staging  →  ECL = PD × LGD × EAD × DF  →  scenario weighting  →  sensitivity
```

On the held-out "reporting-date book" (6,482 loans, €62.5M EAD), the base-scenario
provision is **€10.48M (16.8% coverage)**, the probability-weighted provision is
**€10.80M (+3.1%)** — the gap being a direct demonstration of why IFRS 9 mandates
multiple scenarios: losses are convex in PD, so the ECL of the average scenario is not
the average of the scenario ECLs.

## Data

Kaggle: [`laotse/credit-risk-dataset`](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
— 32,581 consumer loans, 12 columns, binary target `loan_status` (default rate 21.8%).
Data is **downloaded automatically** on first run via `kagglehub`, with a mirrored
fallback. Nothing needs to be placed in `data/` manually.

Data quality is treated as a first-class finding, not a preprocessing footnote: 165
exact duplicates, physically impossible ages (up to 144) and employment lengths (123
years) are removed under documented rules, and missing values are shown to be
*informative* (`person_emp_length` missing → 31.7% default vs 21.6% base) — motivating
in-pipeline imputation with missing-indicators rather than silent fills.

## Repository structure

```
.
├── notebooks/
│   ├── 01_eda_analysis.ipynb     # data-quality audit, cleaning rules, univariate and
│   │                             # interaction structure, statistical association
│   │                             # battery: chi-square/Cramér's V, Mann-Whitney,
│   │                             # mutual information, WoE/Information Value
│   ├── 02_xgboost_model.ipynb    # leakage-free pipeline, imbalance comparison,
│   │                             # RandomizedSearchCV → GridSearchCV, full evaluation,
│   │                             # gain + SHAP importance, and the statistical-vs-model
│   │                             # importance comparison
│   └── 03_ifrs9_ecl.ipynb        # isotonic calibration → PD, SICR staging, LGD/EAD/
│                                 # maturity assumptions, scenario-weighted ECL,
│                                 # sensitivity analysis
├── data/                         # auto-populated, git-ignored
├── requirements.txt
├── LICENSE
└── README.md
```

## Methodology

Same discipline as the rest of the portfolio: preprocessing, imputation and SMOTE live
**inside** cross-validated `imblearn` pipelines (fitted on training folds only); tuning
is a broad `RandomizedSearchCV` (40 candidates × 5 folds) followed by a `GridSearchCV`
built programmatically around the stage-1 winner; evaluation is on an untouched 20%
stratified hold-out.

## Key results

**Model (held-out test set):**

| Metric | Value |
|---|---|
| ROC-AUC | **0.9530** |
| Gini | **0.9060** |
| PR-AUC (AP) | 0.9118 |
| CV ROC-AUC (grid stage) | 0.9454 |
| Calibrated PD: mean vs realized | 0.2214 vs 0.2188 |

**Findings worth reading the notebooks for:**

1. **SMOTE is not free here.** Unlike the churn project, naive SMOTE *costs* about a
   point of ROC-AUC against no correction (0.9314 vs 0.9417) — with marginal signal
   this strong, synthetic interpolation blurs real structure. The search, allowed to
   tune `sampling_strategy` jointly with the tree, ends above the no-correction
   baseline (0.9454). Imbalance correction is a hyperparameter, not a ritual.
2. **Univariate and model importance answer different questions — measurably.** The
   three statistical measures agree with each other almost perfectly (Spearman
   ρ = 0.90–0.98) but only moderately with the model's view (ρ ≈ 0.45–0.62). The
   showcase divergences: `loan_intent` has a "weak" IV of 0.09 yet is the **top**
   feature by mean |SHAP| (its value is entirely interactional), while
   `cb_person_default_on_file` — respectable alone — is the model's *last* feature,
   because prior default is already priced into `loan_grade`.
3. **The grade is the underwriting model.** `loan_grade` is strictly monotone in default
   (A 10% → G 98%) and `loan_int_rate` is priced from it; univariate measures
   double-count this single signal, the multivariate model splits it apart.
4. **The ECL chain passes its structural audits.** Coverage rises monotonically A → G;
   Stage 3 coverage approaches LGD (the PD → 1 limit); and the provision's biggest
   lever is LGD (±10pp ⇒ ∓14% of the provision) — which is why LGD sits at the top of
   the sensitivity disclosure, exactly as it does in bank annual reports.

**IFRS 9 layer (illustrative assumption set, all stated in-notebook):**

| Component | Assumption |
|---|---|
| PD (12m) | isotonic-calibrated model output (point-in-time proxy) |
| SICR → Stage 2 | PD ≥ 2× origination-grade PD, or ≥ 20% backstop |
| Stage 3 | PD ≥ 60% (unlikeliness-to-pay proxy) |
| Lifetime PD | constant hazard, 1−(1−PD₁₂)^M, M = 3–5y by loan intent |
| LGD | 55% / 65% / 75% by housing status |
| EAD | loan amount (fully drawn term loans) |
| Scenarios | base / adverse (PD ×1.4, LGD +5pp) / favourable (PD ×0.7), weighted 50/30/20 |

The closing section of notebook 03 states plainly what this layer is not: the PD is
point-in-time rather than through-the-cycle, staging uses PD proxies in place of
days-past-due triggers, and LGD/EAD are assumption tables rather than workout-data
estimates — with the sensitivity analysis existing precisely because those are the
provision's dominant levers.

## How to run

```bash
git clone https://github.com/ugural87/portfolio_projects.git
cd portfolio_projects/credit-risk
pip install -r requirements.txt
jupyter lab   # run notebooks in order; data downloads automatically on first run
```

Python ≥ 3.10. Note: search objects use `n_jobs=1` deliberately — XGBoost is already
multithreaded internally, and process-based parallelism triggers noisy (harmless) loky
teardown errors on macOS + Python 3.13.

## Roadmap

- Scorecard companion: WoE-binned logistic scorecard as the interpretable challenger,
  with PSI/KS monitoring metrics
- PD term structure from synthetic vintages instead of the constant-hazard assumption
- Macro-linked scenarios (PD as a function of unemployment/rates paths) replacing flat
  multipliers
- LGD modelling if workout-style data can be sourced

## License

MIT — see [LICENSE](LICENSE).
