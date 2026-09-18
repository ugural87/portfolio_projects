# Changelog

## 1.2.3

### Changed
- Decoupled the engineering study guide from application CI. CI no longer regenerates or byte-compares the
  PDF; the guide is produced locally with `make pdf` and does not hard-code test or coverage figures.
- Removed `docs/verification.json`, `tools/update_verification.py`, its PDF-specific test, and
  `make verify-docs`. Application CI now contains only software quality, security, and container checks.

## 1.2.2

### Fixed
- Champion acquisition now fails closed. Registry, authentication, network, artifact-copy, and empty-artifact
  failures stop retraining; bypassing comparison requires the explicit `skip_champion` workflow input.
- Production rollback runs only after the production manifest was successfully applied, so failures during
  checkout, authentication, image tooling, or manifest preparation cannot undo a healthy deployment.
- Staging and production jobs install Kustomize 5.6.0 through a commit-pinned action.
- The Python 3.11 slim base image is pinned to an OCI index digest for reproducible container rebuilds.
- PDF verification metrics are generated from JUnit and coverage JSON. CI regenerates the guide and fails if
  either the metrics file or committed PDF is stale.

## 1.2.1

### Fixed
- **Champion gate was a superiority test.** With a zero margin, the upper-bound rule passed only candidates
  that were significantly *better*; an equally good retrain was blocked with probability ~0.95 (observed:
  5 of 6 in the pipeline, 94.8-96.3% in simulation). The rule is now non-inferiority with a positive margin
  (default 1% of champion MAE, `0` rejected by config). Equal-quality pass rate: 10 of 12 in the pipeline.
- The standard error of the paired difference is cluster-robust over pickup hours; the i.i.d. version
  understated uncertainty under shared traffic conditions. The report adds SE, margin and cluster count.
- A comparison marked `compared` without an upper bound now blocks instead of silently using the point
  estimate; the UCB branch is covered by tests (previously all champion tests took the fallback).
- **Source probe treated network errors as "not published"**, so a transient outage could silently move
  the training window to an older month. Only 403/404 mean unpublished; other failures retry, then raise.
- Target-encoded route feature is described as a smoothed log-duration mean (≈ median only if log-symmetric),
  in ADR 0004 and the PDF guide.

### Added
- ADR 0007, `docs` extra with a hash-locked `requirements-docs.lock`, `make pdf`; guide regenerated.

## 1.2.0

### Fixed
- Dependency audit blockers: `pyarrow` is 23.0.1+ and `pytest` is 9.0.3+; runtime and development
  locks are separate, transitively pinned, and hash-verified.
- Automatic retraining now probes source-object availability and walks backwards within a bounded
  lookback instead of assuming that a two-month calendar lag guarantees publication.
- Rejection rate, out-of-period rate, duplicate rate, and minimum valid rows are release gates.
- Training, validation, champion comparison, offline evaluation, drift predictions, and the API all
  use the same bounded-prediction function.
- Artifact config hashes cover the canonical fully resolved configuration, including inherited base
  values. Data lineage covers every file in the rolling window.
- Runtime compatibility now enforces pandas and joblib as well as Python, scikit-learn, and NumPy.

### Changed
- Scheduled training uses a configurable three-month rolling window and a later chronological
  validation/test tail.
- Champion comparison uses the upper one-sided confidence bound of paired per-row absolute-error
  differences, with a configurable relative tolerance.
- Drift includes pickup zone, drop-off zone, route, unseen-route rate, and prediction distribution.
- Exact duplicate trip rows are removed, counted, and gated under a documented policy.
- Third-party GitHub Actions are pinned to full commit SHAs.
- Feature contract 1.2.0.

### Added
- `requirements-dev.lock`, real-data benchmark protocol, and ADR 0006 documenting the v1.2 controls.

## 1.1.0

### Fixed
- **Prometheus metrics with multiple workers.** The image ran two uvicorn workers while
  `prometheus_client` keeps per-process state, so successive scrapes alternated between workers'
  counters (observed 17 → 3 → 17). Now one worker per container; scale via replicas.
- **Manual retraining with a one-digit month.** `download` zero-padded the month, the train step did
  not (`2025-1.parquet` not found). Period resolution moved into `taxi-ml resolve-period`.
- **Out-of-month rows in monthly files** were not filtered and landed at the extremes of the
  chronological split, corrupting the test window and lineage periods. Rows outside the source month
  are now removed and reported.
- **Drift report hid broken data.** NaN and out-of-range values were dropped by `np.histogram` and the
  rest renormalised (20% broken rows → PSI 0.002). Invalid rates are now reported per feature and can
  mark the batch `critical`.
- **Train/serve library skew.** Training and the image resolved dependencies independently. Added
  `requirements.lock`, recorded runtime versions in metadata, and a load-time compatibility check
  (mismatch → live but not ready).
- Kustomize overlays referenced `staging`/`production` tags that were never pushed. Overlays are now the
  deployment path, with the digest pinned by `kustomize edit set image`.
- 422 contract violations are counted (`status="invalid"`); request logs include the status code.
- Any artifact load failure now yields not-ready instead of a crash.
- Removed the unused `serving.workers` setting and `tools/generate_pdf.py` (reinstated in 1.2 as the
  engineering-guide generator).
- Zones 264/265 (Unknown / Outside NYC) are rejected in training and by the API.

### Changed
- Zones and PU-DO routes are target-encoded (cross-fitted) instead of used as ordinal numbers (ADR 0004).
- Boosting iterations are selected on the chronological validation segment (ADR 0005).
- Release gate compares the candidate with the production champion on the same test window (ADR 0005).
- Gate `max_degradation_vs_baseline: -0.05` renamed to `min_improvement_vs_baseline: 0.05` (same rule).
- Synthetic data uses latent zone geography instead of the zone-ID difference.
- ADR 0003 states the estimand: the conditional median.
- Feature contract 1.1.0.

### Added
- `taxi-ml evaluate` for out-of-period scoring with error breakdowns.
- `taxi-ml drift --fail-on-critical`, CI container smoke test, k8s `/tmp` emptyDir and scrape annotations.
