# ADR 0003: HistGradientBoosting is the production candidate

Status: accepted (rationale for the target transform revised in v1.1)

## Decision

Use a median DummyRegressor as the mandatory baseline and a scikit-learn HistGradientBoostingRegressor
with an absolute-error objective as the candidate. Train on `log1p(duration)` and invert with `expm1`.

## Rationale

The project demonstrates production ML engineering rather than model novelty. The selected estimator
handles nonlinear tabular effects, trains quickly, has a small dependency footprint, and packages with
the feature transformer as one artifact.

## What is being estimated

Quantiles commute with monotone maps, so minimising absolute error in log space yields
`log1p(median Y | x)`; after `expm1` the model returns the conditional median. The transform therefore
does not change the estimand, and tail robustness comes from the L1 loss itself. The transform is kept
because duration effects are largely multiplicative (congestion scales every route), and in log space
they become additive, which suits an additive ensemble of trees.

## Consequences

- Mean bias on right-skewed durations is expected to be negative; it is monitored, not gated.
- The prediction is appropriate for "typical trip" uses (ETA display). Use cases that aggregate
  predictions (fleet capacity) need a mean estimator: a sum of medians is not the median of the sum.
