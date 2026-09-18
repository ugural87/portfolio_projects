# ADR 0001: Predict duration from trip-start information

Status: accepted

## Context

The TLC record contains fields produced after a trip ends, including drop-off time, metered distance,
fare components, payment type, and totals. Using them would improve offline scores while making the
online system impossible or dishonest.

## Decision

The API accepts pickup time, pickup zone, intended drop-off zone, and passenger count. Drop-off time
is used only to construct the training label. No fare, realized distance, payment, or drop-off-time
field enters the model pipeline.

## Consequences

The model is weaker than a retrospective estimator but matches the stated decision point. Offline and
online feature transformations share one scikit-learn Pipeline, preventing training-serving skew.

