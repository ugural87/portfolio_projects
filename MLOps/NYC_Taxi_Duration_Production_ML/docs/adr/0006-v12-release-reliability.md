# ADR 0006: Availability-aware rolling training and unified release behavior

Status: accepted (v1.2)

## Context

A fixed calendar lag can point at an unpublished TLC file. One-month training overfits recent seasonality,
quality statistics were reported but did not block release, and raw training predictions differed from the
bounded predictions served by the API. The drift report also omitted the categorical route signals that carry
most of the model's information.

## Decision

- Probe the candidate source object and walk backwards for at most `source_lookback_months`.
- Train scheduled candidates on `training_window_months` separately validated monthly files, then split the
  merged rows chronologically.
- Gate rejection, out-of-period, exact-duplicate, and valid-row volume before artifact creation.
- Route every model score through `predict_duration`, which applies the production bounds.
- Compare candidate and champion on identical rows using the one-sided confidence bound of paired absolute
  error differences. Permit only the configured relative regression tolerance.
- Monitor numeric, zone, route, unseen-route, and prediction drift.
- Hash resolved configuration and the checksum manifest of every training source file.

## Consequences

Retraining fails closed when publication or quality expectations are not met. Seasonal coverage improves at
the cost of more download, compute, and storage. (v1.2.1 correction) As first shipped, the confidence
gate used a zero margin, which made it a superiority test: a candidate exactly as good as the champion was
blocked with probability about 0.95 (0.5 for the v1.1 point-estimate rule). ADR 0007 replaces it with a
non-inferiority rule and an hour-clustered standard error. The normal approximation should still be
replaced with a block bootstrap if strong temporal
autocorrelation is observed. Exact duplicate removal avoids inventing a trip identity; near-duplicates remain.
