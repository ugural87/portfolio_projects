# Executed Results

All figures and numbers below were produced by `scripts/run_pipeline.py` from the canonical ULB/OpenML raw dataset.

## Data

- Raw transactions: 284,807
- Modeling transactions after exact-row deduplication: 283,726
- Exact duplicates removed: 1,081
- Modeling frauds: 473 (0.1667%)

## Selected predictive model

- Candidate: `deep_mlp__focal`
- Selection metric: chronological validation PR-AUC
- Strategy: `focal`
- Final holdout PR-AUC: 0.7925
- Final holdout PR-AUC 95% interval: [0.6902, 0.8923]

## Deep-learning challenger

- Selected loss: `focal`
- Validation PR-AUC: 0.8010
- Selected as overall predictive winner: True

## Frozen decision policy

- Policy selected before final test: `f1_threshold`
- Final test transactions: 42,557
- Alerts: 35
- Precision: 100.00%
- Fraud count recall: 67.31%
- Fraud amount capture: 56.22%
- Gross prevented loss: 3,194.38 CU
- Net savings: 3,054.38 CU
- Bootstrap 95% interval: [1,263.01, 5,471.84] CU
- Illustrative savings per million transactions: 71,771.60 CU

These are scenario estimates under the documented cost assumptions, not realized savings from an identified bank.

## Semi-synthetic causal audit

- True constructed ATE: 11.8711 CU
- Cross-fitted AIPW ATE: 11.5437 CU
- IPTW effective sample size: 39,854
- Full-refit bootstrap repetitions: 30

The causal section validates estimators against constructed potential outcomes. It does not claim that the public data identify a real review effect.
