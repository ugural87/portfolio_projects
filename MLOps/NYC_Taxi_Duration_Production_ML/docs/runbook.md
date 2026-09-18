# Operations runbook

## Symptoms and first checks

1. `/health/live` fails: container or process failure. Inspect platform events and application logs.
2. `/health/ready` returns 503: artifact missing, corrupt, or written by a different Python/scikit-learn/NumPy/pandas/joblib
   version (`model_load_failed` log line names the mismatch). Do not route traffic; rebuild from the lock.
3. Error-rate alert: correlate request IDs, HTTP status labels, schema errors, and recent deployments.
4. Latency alert: inspect replica saturation, CPU/memory throttling, and prediction histogram volume.
5. Drift `critical`: feature/categorical/prediction PSI exceeds its limit, invalid-value rate exceeds
   `max_invalid_rate`, or unseen-route rate exceeds `max_unseen_route_rate`. The report
   separates the two. A high invalid rate is an upstream data problem, not model drift: fix ingestion
   first. For PSI, quarantine automatic promotion, inspect changed fields and compare performance once
   labels arrive.
6. MAE breach after labels arrive: stop promotion, retain champion, train a challenger, and diagnose by
   time, route, and zone.

## Rollback

The production deployment uses an immutable image digest. `kubectl rollout undo deployment/taxi-duration-api`
restores the previous ReplicaSet. Confirm readiness and execute `scripts/smoke_test.sh` against production.
The workflow invokes rollback only if the production manifest was successfully applied; failures before that
point leave the healthy deployment untouched. Rollback does not delete the failed image or model; retain both
for incident analysis.

## Metrics look erratic

Counters that jump back and forth between scrapes mean more than one process is serving one `/metrics`
endpoint. The image runs one worker per container by design; check that no override added `--workers`.

## Model promotion

Only artifacts produced by the retraining workflow are eligible. Automatic runs first probe TLC publication,
fall back month-by-month within the configured bound, download the complete rolling window, and fail closed
if none is available. The workflow compares each candidate
against the image tagged `production` and moves that tag only after a healthy production rollout. Run with
`skip_champion` only for the first release or a deliberate runtime-lock upgrade, and record why. Without that
explicit input, any production-image pull or artifact extraction failure stops the release; infrastructure
failure is never interpreted as absence of a champion. A human reviews data-quality results,
time ranges, baseline comparison, validation/test metrics, drift, and metadata. Promotion reuses the exact
tested digest. Never rebuild between staging and production. A data-quality gate failure is not waived by
good MAE; investigate source publication/schema behavior first.
