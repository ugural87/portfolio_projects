# Changelog

## 0.1.2 — v7.1 checkpoint correctness and progress

- Preserved the v7 model and feature architecture byte-for-byte.
- Changed all four event-model checkpoints to select the minimum validation joint loss; the same
  metric now drives early stopping.
- Restored and re-evaluated the selected weights before checkpoint serialization, and recorded the
  selected epoch/loss, minimum delta, patience and completed epochs in metadata.
- Added terminal/Jupyter `tqdm` progress for data ingestion, extraction/cache auditing, model
  training, bootstrap tests and notebook post-processing. `US10Y_PROGRESS=0` disables bars.
- Rebased legacy absolute minutes paths safely to the current project directory after filename
  verification, making the executed package portable across machines.
- Added real-data checkpoint-restoration tests and an executed-v7 release verifier.
- Re-ran the full real-data study: 8 folds, 32 event checkpoints and 178 OOS predictions.
- Added a report-ready, layered software/ML architecture SVG plus editable Mermaid source.

## 0.1.1 — HTTP content negotiation

- Split retrying HTTP sessions by provider: FRED requests explicitly accept JSON, while Federal
  Reserve discovery/download requests accept HTML, XHTML and PDF content.
- Kept the retry policy, backoff and request headers centralized and backward compatible.

## 0.1.0 — modular master package

- Split the validated notebook workflow into importable data, LLM, feature, model, training,
  validation, evaluation and reporting modules.
- Preserved the official-Fed discovery repairs: meeting-span date validation, malformed historical
  heading handling and exclusion of the known 2003-09-15 link mismatch.
- Kept FRED policy-rate facts authoritative; minutes-derived targets remain diagnostics only.
- Preserved the 14-feature Luna contract, sentence-ID grounding, local evidence hydration,
  pipeline/inference signatures, quarantine and durable paid-cost ledger.
- Replaced literal/fuzzy evidence dependence with deterministic sentence-catalog selection. Fed
  ellipses are normalized only when constructing the local catalog.
- Added a two-channel semantic token (`score`, `confidence`) plus an explicit availability mask.
- Added bidirectional four-head cross-attention and exportable per-head attention matrices.
- Corrected transfer leakage from the earlier monolithic design: FOMC models load only a dedicated
  price backbone whose labels end before the first FOMC meeting. The standalone price benchmark is
  explicitly separate.
- Added four paired walk-forward models: price-only, rate-only, shuffled-text and full fusion.
- Added per-fold split conformal calibration with a minimum of 19 calibration observations.
- Added hard checks for split overlap, quantile ordering, attention masks, frozen parameters,
  checkpoint purpose and checkpoint information cutoff.
- Added resumable runtime logs, publication tables, static figures, an interactive architecture
  graph and three thin notebooks.

## Compatibility note

Passing pilot records are accepted only if document hashes, model ID, schema/pipeline signature and
inference-policy signature all match. Old records are never silently reused under a changed contract.
