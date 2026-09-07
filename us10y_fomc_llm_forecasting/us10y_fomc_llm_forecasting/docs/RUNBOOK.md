# Runbook

## 1. Create the environment

```bash
conda activate torch_env
cd /path/to/us10y_fomc_fusion
python -m pip install -e ".[dev]"
python scripts/check_environment.py
```

Python 3.11 is the tested target. PyTorch can use Apple MPS, CUDA, or CPU automatically.
The release verification was also completed with Python 3.12 and CPU PyTorch.

## 2. Store credentials safely

Copy `.env.example` to `.env` and save it as plain text:

```text
FRED_API_KEY=your_fred_key
OPENAI_API_KEY=your_openai_project_key
```

Do not use a rich-text TextEdit document. The loader rejects RTF. `.env` is ignored by Git,
keys are never printed, and audit logs contain only a short SHA-256 fingerprint.

## 3. Download official data

Keep all paid switches `false`, then run:

```bash
python scripts/download_data.py
python scripts/run_preflight.py
```

This downloads the FRED Treasury panel and official Federal Reserve minutes to local files.
It also enforces document hashes, year coverage, canonical URL dates and three known policy-rate
anchors. The minutes rate parser is diagnostic only; FRED remains authoritative.

## 4. Validate model access without inference

Set the following in `configs/runtime.toml`:

```toml
run_openai_preflight = true
confirm_api_key = true
run_paid_extraction = false
```

Then rerun `python scripts/run_preflight.py`. A model retrieve call is made, but no inference is
requested.

## 5. Promote the passing pilot and run paid extraction

The validated five-pair pilot can be supplied without paying for it again:

```bash
python scripts/run_luna_extraction.py --pilot /path/to/luna_paid_pilot.jsonl
```

Before the production call, set:

```toml
run_paid_extraction = true
confirm_paid_extraction = true
max_new_pairs = 0
```

and set a positive `hard_run_budget_usd` in `configs/fomc_extraction.toml`. The ledger reserves
the worst-case cost before each request, reconciles actual usage afterward, caches accepted
records, and quarantines incomplete, schema-invalid or insufficiently grounded results. Restarting
is safe: qualified cached pairs are not called again.

Run:

```bash
python scripts/run_luna_extraction.py --pilot /path/to/luna_paid_pilot.jsonl
python scripts/run_preflight.py
```

Do not continue until `production_extraction_complete` and `go_for_training` are true.

## 6. Train models

First set:

```toml
train_historical_backbone = true
```

Run:

```bash
python scripts/train_historical_backbone.py
```

The saved checkpoint contains its information boundary. Loading fails unless its last training
label precedes the first FOMC meeting.

For the four-model study, set:

```toml
train_fusion_models = true
run_walk_forward = true
```

Run:

```bash
python scripts/run_walk_forward.py
```

The terminal and notebooks show nested `tqdm` progress for folds, models and epochs. Each model is
selected by minimum validation joint loss; balanced accuracy is diagnostic. The same loss controls
early-stopping patience. The selected weights are restored and re-evaluated before saving. For CI
or plain log files, run with `US10Y_PROGRESS=0`.

Every fold reserves at least 19 calibration meetings, applies split conformal correction inside
the fold, and evaluates all four controls on identical out-of-sample meetings. The final twelve
months remain sealed unless `evaluate_final_holdout` is deliberately enabled in a future analysis.

The standalone daily price benchmark is optional and is never used as the FOMC backbone. To train
it, set `train_price_benchmark = true` and run `python scripts/train_price_benchmark.py`.

## 7. Build evidence and figures

```bash
python scripts/build_report.py
```

Use `notebooks/03_postprocess.ipynb` for interactive Plotly charts and inspection tables.
Use `docs/architecture.svg` directly in the final report; `docs/architecture.mmd` is its editable
source. The optional legacy Plotly view remains available through
`python scripts/build_architecture_html.py`.

## Recovery

- Paid requests are resumable from `data/external/fomc/extraction_cache`.
- Rejected pairs remain in `data/external/fomc/quarantine` and are not retried unless both retry
  switches are explicitly enabled.
- Checkpoints, predictions and attention tables are versioned by stage/fold under `outputs`.
- Runtime progress is appended as JSONL under `outputs/runtime`; use `tail -f` while training.
- `scripts/verify_v7_1_release.py --baseline /path/to/executed-v7` verifies fold identity, protected
  source hashes, prediction contracts and all 32 checkpoint selections.
