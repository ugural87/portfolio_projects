# Bank Customer Churn — Prediction, Calibration and Campaign Economics

End-to-end churn modelling on the classic Kaggle bank churn dataset (10,000 customers),
built as a **decision system rather than a classifier**: the output is not a label but a
calibrated probability feeding an explicit profit-maximizing contact policy.

## Why this framing

Churn is usually presented as binary classification and evaluated with accuracy or F1.
That framing silently fixes the decision threshold at 0.5 — a choice no bank would make.
With a contact cost of €50, a 30% save rate and €2,000 preserved customer value, the
break-even condition

```
p · (0.30 × 2000) > 50   ⟹   p* ≈ 0.083
```

says the economically optimal policy contacts every customer above ~8% churn probability.
On the held-out test set this repository's calibrated model reproduces that number
empirically (profit-maximizing threshold **0.075** vs closed-form **0.083**) — and the
profit gap is material: **€168.4k** at the optimal threshold vs **€109.6k** at the naive
0.5 threshold vs €144.2k contacting everyone. The 0.5-threshold classifier is worse than
having no model at all. That is the thesis of this project.

## Data

Kaggle: [`shrutimechlearn/churn-modelling`](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
— 10,000 bank customers, 14 columns, binary target `Exited` (base churn rate 20.37%).
Data is **downloaded automatically** on first run via `kagglehub`, with a byte-identical
GitHub mirror as fallback. Nothing needs to be placed in `data/` manually.

## Repository structure

```
.
├── notebooks/
│   ├── 01_eda_analysis.ipynb                # detailed EDA: univariate, interactions,
│   │                                        # the three structural findings
│   ├── 02_logistic_regression.ipynb         # linear baseline: SMOTE, two-stage
│   │                                        # hyperparameter search, full evaluation
│   ├── 03_xgboost.ipynb                     # tuned XGBoost: same protocol, SHAP
│   ├── 04_lightgbm.ipynb                    # tuned LightGBM + series-closing
│   │                                        # three-model comparison
│   ├── 05_decision_analysis.ipynb           # calibration, profit-optimal threshold,
│   │                                        # cost sensitivity, business proposals
│   └── 06_causal_propensity.ipynb           # causal analysis: propensity scores,
│                                            # IPW / matching / g-computation
├── data/                                    # auto-populated, git-ignored
├── requirements.txt
├── LICENSE
└── README.md
```

Each model notebook follows an **identical evaluation protocol**, so results are directly
comparable: leakage-free `imblearn` pipeline (SMOTE applied only inside training folds) →
imbalance-strategy comparison → broad `RandomizedSearchCV` → refined `GridSearchCV`
built programmatically around the stage-1 optimum → held-out ROC/PR curves, confusion
matrices at multiple thresholds, classification report → interpretability
(coefficients/SHAP).

## Key findings

**Structure of the data (EDA):**

1. **Geography–balance entanglement.** German customers churn at 32.4% vs ~16.5% in
   France/Spain — and Germany has essentially no zero-balance accounts while
   France/Spain have ~45–50%. A raw balance coefficient partly proxies nationality;
   the pipeline adds an explicit `zero_balance` flag and a Germany×Balance interaction.
2. **Non-monotonic product effect.** Two products is the safest state (7.6% churn);
   three or four products means near-certain churn (82.7% / 100%). A U-shape no single
   linear coefficient can represent — `NumOfProducts` enters the linear model as a
   categorical, and trees capture it natively.
3. **Age is the strongest driver**, and non-linearly so: churners average 44.8 years vs
   37.4 for retained, with risk peaking around 50–60.

**Model results (held-out 20% test set, stratified):**

| Model | CV ROC-AUC (5-fold) | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|
| Logistic Regression (nb 02) | 0.8388 | 0.8389 | 0.6625 |
| XGBoost (nb 03) | 0.8657 | **0.8700** | **0.7228** |
| LightGBM (nb 04) | 0.8654 | 0.8681 | 0.7217 |

All three: SMOTE inside the CV pipeline, two-stage RandomizedSearchCV → GridSearchCV
tuning, identical split and preprocessing.

**Methodological results:**

- **Rebalancing does not move AUC for the linear model** (0.8385 → 0.8376 with SMOTE):
  reweighting shifts the intercept, not the ranking. The gain is recall at a fixed 0.5
  threshold (0.415 → 0.732) at the cost of probability quality (Brier 0.111 → 0.157) —
  i.e. resampling and threshold choice are one decision, not two.
- **Calibration closes the loop with economics (nb 05):** after calibrating the tuned
  XGBoost, the empirical profit-maximizing threshold (0.075) sits at the closed-form
  break-even (0.083) within the flatness of the profit curve, and the sensitivity grid
  shows the optimal threshold staying in a 0.04–0.17 band across the entire plausible
  cost-parameter space — always far below 0.5. At the recommended threshold the campaign
  reaches 93% of true churners at 282% ROI on budget.
- **The linear model's ceiling is representational, not a tuning artifact:** a 40-candidate
  randomized search followed by a refined grid moved CV ROC-AUC by 0.0000 — the top-5
  configurations sit within a 0.0003 band. Tuned XGBoost converts that headroom:
  +3.1 points ROC-AUC and +6.0 points PR-AUC over the tuned linear model on the same
  split — and its SHAP dependence plots show the gap coming precisely from the
  non-linear age effect and the U-shaped product effect identified in EDA.
- **For trees, shallow-and-slow won:** the two-stage search settled on depth-3 trees
  with a small learning rate (280 rounds), and `scale_pos_weight`/SMOTE again left
  ROC-AUC essentially unchanged (0.863 ± 0.001 across all three strategies) —
  imbalance correction is a threshold decision, not a ranking decision, for this data.
- **XGBoost and LightGBM land within noise of each other** (ΔAUC = 0.002 against a CV
  fold std of ±0.01): with identical features and an identical two-stage search, the
  choice between well-tuned gradient boosters is an engineering preference here, not a
  statistical result. The series-closing comparison lives in notebook 04.
- **Association is not a lever (nb 06):** activity is the strongest protective factor in
  every model (adjusted odds ratio 0.33), and a propensity-score analysis — IPW,
  caliper matching, g-computation, all with post-weighting max |SMD| ≈ 0.01 — puts the
  effect of activity at −12 to −13 pp of churn consistently across estimators. The
  notebook argues plainly why this is an upper bound (inactivity is partly a *symptom*
  of churn intent) and why the decisive evidence is an A/B-tested activation program.

## How to run

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt
jupyter lab   # run notebooks in order; data downloads automatically on first run
```

Python ≥ 3.10. Note: `cross_validate`/search objects use `n_jobs=1` deliberately —
process-based parallelism adds nothing at this data size and triggers noisy (harmless)
loky teardown errors on macOS + Python 3.13.

## Roadmap

- SMOTENC variant (design-correct oversampling for mixed categorical/numeric data)
- Uplift modelling on experiment data — the causal notebook's endpoint is an A/B-tested
  activation program; uplift models would then target the *persuadables*
- Temporal validation & drift discussion (this dataset is a snapshot; a production
  system needs out-of-time splits)
- Fairness note: gender enters the model; deployment requires a documented policy
  decision on protected attributes

## License

MIT — see [LICENSE](LICENSE).
