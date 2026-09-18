# ADR 0007: Champion comparison is a non-inferiority test with hour-clustered errors

Status: accepted (v1.2.1), corrects ADR 0006

## Problem

The question a release gate asks is: "is the candidate at least as good as what is serving?" For paired
per-row differences `d_i = |y_i - candidate_i| - |y_i - champion_i|` (positive = candidate worse) with mean
`d̄` and standard error `SE`, v1.2 blocked when `d̄ + z·SE > m`, with margin `m = 0`. Passing then requires
`d̄ < -z·SE`, i.e. statistically significant *improvement*. That is a superiority test. If the two models are
equally good (`E[d] = 0`), the pass probability is `P(d̄ < -z·SE) = 1 - Φ(z) = 0.05` at 95% confidence,
independent of sample size. Measured: 5 of 6 equal-quality retrains blocked in the pipeline; 94.8% (n = 3,000)
and 96.3% (n = 300,000) blocked in simulation. The consequence is a frozen champion whenever the data brings
no provable gain, which is the normal case for a stable process.

## Decision

1. **Non-inferiority.** Keep the upper bound but require a positive margin: block when
   `d̄ + z·SE > m`, `m = max_regression_vs_champion × champion MAE`, default 1%, validated `0 < m < 20%`.
   The burden of proof stays on the candidate (promotion is the risky action), but "as good as" now passes.
   As the test set grows, `SE → 0` and the rule converges to "no worse than the margin".
2. **Cluster-robust SE by pickup hour.** Trips in the same hour share traffic, so their `d_i` are
   correlated and the i.i.d. SE understates uncertainty. With clusters `g` of the test rows,
   `Var(d̄) = G/(G-1) · Σ_g (Σ_{i∈g}(d_i - d̄))² / n²`. Fewer than two clusters is a failed comparison.
3. The report records `mae_delta`, its SE, the upper bound, the margin in minutes, and the cluster count.

## Evidence

Same data, seed-only differences, 20k synthetic rows (51 hourly clusters in the test window): 10 of 12
retrains pass; the two blocked have the largest mean deltas (+0.054, +0.035 min against a 0.072 min margin).
Unit simulation with an hour-level shared component: ~83% of equal candidates pass versus ~5% under the zero
margin; a candidate 0.5 min worse per trip is blocked.

## Trade-offs

The margin is a business tolerance, not a statistical quantity; 1% of MAE (~4 s on a 7-minute MAE) should be
revisited with the real-data benchmark. Hour clusters assume independence across hours; day-level or block
bootstrap inference remains the next step if cross-hour correlation is material.
