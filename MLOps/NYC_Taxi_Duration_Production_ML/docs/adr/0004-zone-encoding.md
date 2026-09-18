# ADR 0004: Target-encode zones and routes

Status: accepted (v1.1)

## Context

TLC `LocationID`s are assigned in the alphabetical order of zone names. In v1.0 they entered the model as
numbers, which asserts an ordering that has no geographic meaning; trees can still isolate zones, but only
by spending many threshold splits, and the pickup-drop-off interaction (the route) is what drives duration.
HistGradientBoosting's native categorical support is capped at 255 categories, below the 263 routable zones.

## Decision

Build `route_id = PU * 1000 + DO` and target-encode route, pickup zone and drop-off zone with
`sklearn.preprocessing.TargetEncoder`:

- cross-fitted during `fit` (5-fold, out-of-fold values), so no training row is encoded with its own label;
- full training statistics at inference;
- empirical-Bayes smoothing (`smooth="auto"`) shrinks sparse routes toward the global mean, while the zone
  encodings carry the marginal zone effects;
- unseen routes receive the training mean; `metrics.json` reports error for seen vs unseen routes.

The encoder lives inside `TransformedTargetRegressor`, so it encodes `log1p(duration)`, consistent with
ADR 0003. `model.zone_encoding: ordinal` keeps the v1.0 arm for comparison.

## Evidence so far

On the synthetic generator (latent zone geography, 20k rows) test MAE fell from 10.53 to 7.21 minutes.
This only shows that the mechanism works when geography matters; the real-data comparison must be run on
TLC files and recorded here before the decision is considered validated.

## Alternatives considered

Static zone-centroid coordinates and centroid distance (from the TLC zone shapefile) are leakage-safe and
cheap, and remain a valid next challenger. Target encoding was chosen first because it learns realised route
durations directly instead of a straight-line proxy for them. Precisely, it learns the smoothed mean of
log1p(duration) per route, i.e. a geometric-mean-type centre; that equals the route median only when
log-duration is symmetric within the route. The model's final estimand is still the conditional median
(L1 loss, ADR 0003); this note concerns only what the encoded feature represents.
