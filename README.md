[README.md](https://github.com/user-attachments/files/30325989/README.md)
# Data Science Portfolio

A collection of end-to-end data science projects built around one idea: **models are not
the deliverable — decisions are.** Every project runs the full arc from exploratory
analysis through rigorously tuned models to the decision layer (cost-aware thresholds,
business proposals) and, where the data allows, to causal analysis of the levers the
business actually controls.

My background is ~13 years of quantitative modelling — from continuum mechanics and
constitutive modelling to financial markets — and these projects apply that mindset to
applied data science: explicit assumptions, leakage-free protocols, calibrated
probabilities, and honest statements of what the data can and cannot support.

## Design principles

Every project in this portfolio follows the same standards, applied in the form the
project type demands:

- **Reproducible by one command** — data is downloaded automatically on first run
  (Kaggle via `kagglehub`, with mirrored fallbacks); no manual file placement.
- **Leakage-free by construction** — no information crosses from evaluation data into
  training. In supervised ML this means preprocessing and resampling live *inside*
  cross-validated pipelines; in time series it means walk-forward splits and strict
  no-lookahead feature construction; in NLP it means fitting vectorizers and embeddings
  on training folds only.
- **Fair model comparison: the model is the only variable** — compared models share the
  same split, features and tuning budget. In the classical-ML projects this takes the
  form of an identical two-stage search per model (broad `RandomizedSearchCV` → refined
  `GridSearchCV`, as in the churn series); in the deep learning projects, a shared
  training/validation regime with the classical models standing as baselines that any
  deep architecture must demonstrably beat.
- **Probabilities over labels, wherever a probability is the product** — classification
  outputs are calibrated and audited with reliability diagrams, and forecasts are
  evaluated with proper scoring rules, because real decisions are priced on
  probabilities, not argmax labels.
- **Every project ends at the decision it informs** — cost matrices and profit-optimal
  thresholds where economics are explicit; alert budgets in fraud; action-per-segment
  logic in segmentation; sensitivity analysis of the assumptions in all of them.
- **Association vs causation stated honestly** — predictive importance is never sold as
  a lever; causal claims get causal machinery (propensity scores, weighting, matching)
  and plainly stated limitations.

## Projects

| # | Project | Domain | Status |
|---|---------|--------|--------|
| 1 | [Bank Customer Churn](./bank-churn/) | Retail banking / retention | ✅ Complete |
| 2 | [Credit Risk Modelling](./credit-risk/) | Retail credit / PD estimation | ✅ Complete |
| 3 | [Customer Segmentation](./customer-segmentation/) | Marketing analytics / unsupervised | ✅ Complete |
| 4 | Fraud Detection | Payments / anomaly detection | Planned |
| 5 | Classical Time Series Analysis | Sensor & financial data / statistical TS | Planned |
| 6 | Deep Learning Time Series Forecasting | Markets / sequence modelling | Planned |
| 7 | NLP & LLM Track | Text / retrieval / generation | Planned |

---

### 1. Bank Customer Churn ✅

Churn as a *decision system*, not a classification exercise. Six notebooks, each
building on the last:

| Notebook | Content |
|---|---|
| `01_eda_analysis` | Detailed EDA — univariate structure, interactions, and the three structural findings that drive every downstream choice |
| `02_logistic_regression` | The linear yardstick: SMOTE inside the CV pipeline, two-stage hyperparameter search, full evaluation suite (ROC/PR, confusion matrices, threshold analysis), odds-ratio interpretation |
| `03_xgboost` | Same protocol, XGBoost — SHAP shows the gap over the linear model coming exactly from the non-linearities found in EDA |
| `04_lightgbm` | Same protocol, LightGBM — plus the series-closing three-model comparison |
| `05_decision_analysis` | Calibration, cost-matrix-derived optimal contact threshold, sensitivity analysis over the economic assumptions, campaign sizing, five concrete business proposals |
| `06_causal_propensity` | From association to causation: propensity-score analysis (IPW, caliper matching, g-computation) of the activation lever, with balance diagnostics and honest limitations |

### 2. Credit Risk Modelling ✅

Probability-of-default modelling carried all the way to the number a bank actually
books: a staged, scenario-weighted **IFRS 9 expected credit loss** with every assumption
stated and stress-tested. Three notebooks:

| Notebook | Content |
|---|---|
| `01_eda_analysis` | Data-quality audit with documented cleaning rules, informative-missingness analysis, and a statistical association battery: chi-square/Cramér's V, Mann-Whitney, mutual information, WoE/Information Value |
| `02_xgboost_model` | Leakage-free pipeline (in-pipeline imputation + missing indicators, SMOTE in CV folds), two-stage hyperparameter search, full evaluation (ROC/PR, Gini, confusion matrices), gain + SHAP importance — closing with a measured comparison of statistical vs model importance and where the two rankings diverge |
| `03_ifrs9_ecl` | Isotonic calibration → PD, SICR staging with relative and backstop rules, explicit LGD/EAD/maturity assumptions, probability-weighted scenarios, and a sensitivity disclosure of the provision to its assumption set |

### 3. Customer Segmentation ✅

Unsupervised counterpart to the supervised projects, built around one disciplined
question: *how much structure does this data actually contain?* Four notebooks:

| Notebook | Content |
|---|---|
| `01_eda_rfm` | Transaction-log audit with documented cleaning rules, RFM+ feature construction, skew/log analysis, correlation structure |
| `02_kmeans` | k selection with three instruments — elbow, silhouette curve, and per-cluster silhouette "knife" plots — then segment profiling and one action per segment |
| `03_dbscan` | `min_samples` reasoning and k-distance eps selection, an eps grid mapping over-smoothing against fragmentation, knife-plot audit, and noise analysis that turns out to be the project's most valuable output |
| `04_pca_tsne` | PCA variance, loadings and biplot with named components; t-SNE across a perplexity scan; both clusterings overlaid on both projections |

Three independent methods converge on the verdict that the RFM space is a continuum
rather than a set of islands — so the segments are presented as useful partitions, and
the density method's contribution is reframed as a principled key-account list.

### 4. Fraud Detection

Extreme class imbalance as the central technical challenge: precision-recall-first
evaluation, cost-sensitive thresholds where false positives and false negatives carry
asymmetric and very different costs, anomaly-detection baselines vs supervised
approaches, and the operational questions (alert budgets, queue prioritization) that
make fraud a decision problem rather than a leaderboard metric.

### 5. Classical Time Series Analysis

The statistical foundation the deep models must justify themselves against, built on two
deliberately different datasets: **sensor data** (strong seasonality, clean physical
structure) and **financial return series** (weak mean signal, strong volatility
structure). Content:

- **Structure and diagnostics** — decomposition into trend / seasonality / remainder
  (classical and STL), autocorrelation analysis (ACF/PACF) for identifying
  autoregressive and moving-average terms, stationarity testing (ADF, KPSS) and
  differencing decisions.
- **The ARIMA family** — ARMA → SARIMA for seasonal structure → SARIMAX with exogenous
  regressors; order selection via information criteria against ACF/PACF reasoning,
  residual diagnostics (Ljung-Box) as the acceptance test.
- **Volatility modelling** — ARCH/GARCH on financial returns: volatility clustering,
  conditional heteroskedasticity, and why modelling the second moment is often the only
  honest thing to do with return series.
- **Prophet and smoothing benchmarks** — Prophet and Holt-Winters exponential smoothing
  as pragmatic baselines, with a sober comparison of where they win (multiple
  seasonalities, holidays, missing data) and where they don't.

### 6. Deep Learning Time Series Forecasting

Hybrid deep architectures for financial series: **2D convolutional feature extraction
over multi-channel input windows feeding a Transformer encoder** — CNN layers learn
local temporal/cross-channel patterns, attention layers learn long-range structure —
with **LSTM** models built alongside as the recurrent reference point, so the
convolution+attention hybrid is judged against the architecture it claims to improve on.
The target is **direction classification rather than point estimation**: point forecasts
of financial series systematically overstate what the data supports, so the honest
formulation is directional probability, evaluated with proper scoring rules and
calibration rather than RMSE theater. Full temporal hygiene throughout: walk-forward
validation, strict no-lookahead feature construction, and the classical baselines of
project 5 (SARIMA/GARCH/Prophet) that any deep model must demonstrably beat.

### 7. NLP & LLM Track

Three connected builds moving up the abstraction ladder:

- **Recommendation system** — content-based recommendation over text: embedding
  construction, similarity retrieval, and evaluation beyond accuracy (coverage,
  diversity, cold-start behavior).
- **Sentiment classification with LLMs** — classical baselines (TF-IDF / embedding
  pipelines into gradient boosting) and a **BiLSTM + attention** architecture — used in
  earlier work and carried in here as the pre-transformer deep baseline — against
  fine-tuned transformer and LLM-based classifiers, with error analysis on where each
  paradigm wins and at what cost.
- **RAG system** — retrieval-augmented generation end to end: chunking and embedding
  strategy, vector retrieval, generation with grounding, and evaluation of both
  retrieval quality and answer faithfulness.

---

## Tech stack

- **Core:** Python · pandas · NumPy · SciPy
- **Machine learning:** scikit-learn · imbalanced-learn · XGBoost · LightGBM · SHAP
- **Deep learning:** PyTorch (CNNs, LSTMs, Transformer encoders, custom architectures)
- **NLP / LLM:** Hugging Face Transformers · sentence-transformers · gensim
- **Time series:** statsmodels (ARIMA/SARIMAX, decomposition, diagnostics) · pmdarima
  (auto-ARIMA order search) · arch (ARCH/GARCH) · Prophet
- **Visualization:** matplotlib · seaborn · Plotly
- **Apps & dashboards:** Streamlit
- **Tooling:** Jupyter · Git

Each project folder carries its own `requirements.txt` and README with full results and
reproduction instructions.

## Structure convention

```
portfolio_projects/
├── README.md                          # this page — the portfolio map
│
├── bank-churn/                        # churn as a decision system
│   ├── notebooks/
│   │   ├── 01_eda_analysis.ipynb      # EDA: three structural findings
│   │   ├── 02_logistic_regression.ipynb   # linear baseline, SMOTE, two-stage search
│   │   ├── 03_xgboost.ipynb           # same protocol, XGBoost, SHAP
│   │   ├── 04_lightgbm.ipynb          # same protocol + three-model comparison
│   │   ├── 05_decision_analysis.ipynb # calibration, profit threshold, proposals
│   │   └── 06_causal_propensity.ipynb # IPW / matching / g-computation
│   ├── data/                          # auto-downloaded, git-ignored (.gitkeep only)
│   ├── requirements.txt               # project-specific pinned minimums
│   ├── LICENSE                        # MIT
│   └── README.md                      # results, findings, roadmap
│
├── credit-risk/                       # PD model + IFRS 9 ECL layer
│   ├── notebooks/
│   │   ├── 01_eda_analysis.ipynb      # data-quality audit, chi2/Cramér's V/MI/IV battery
│   │   ├── 02_xgboost_model.ipynb     # tuned XGBoost, SHAP, statistical-vs-model importance
│   │   └── 03_ifrs9_ecl.ipynb         # calibrated PD, staging, LGD/EAD, scenario ECL
│   ├── data/
│   ├── requirements.txt
│   ├── LICENSE
│   └── README.md
│
├── customer-segmentation/             # unsupervised: how much structure is really there?
│   ├── notebooks/
│   │   ├── 01_eda_rfm.ipynb           # transaction-log audit, RFM+ construction
│   │   ├── 02_kmeans.ipynb            # elbow, silhouette curve, knife plots, actions
│   │   ├── 03_dbscan.ipynb            # k-distance eps selection, noise analysis
│   │   └── 04_pca_tsne.ipynb          # PCA loadings/biplot, t-SNE perplexity scan
│   ├── data/
│   ├── requirements.txt
│   ├── LICENSE
│   └── README.md
│
└── ...                                # each subsequent project follows the same layout
```

Every project folder is self-contained: its own pinned `requirements.txt`, its own MIT
`LICENSE`, its own README carrying the results, and a git-ignored `data/` directory that
the first notebook populates automatically. Notebooks are numbered in reading order and
committed **with their outputs**, so the analysis is readable on GitHub without running
anything.

Results, figures and detailed findings live in each project's own README — this page is
the map, not the territory.
