# US10Y–FOMC Cross-Attention Research Pipeline

This package forecasts the five-business-day change in the US 10-year Treasury yield around
scheduled FOMC meetings. It combines:

- a 2D CNN over causal daily yield-curve features;
- a Transformer over daily price tokens;
- an authoritative FRED rate-decision token;
- 14 individually masked, sentence-grounded minutes features extracted by `gpt-5.6-luna`;
- bidirectional event/price cross-attention;
- ordered quantile forecasts, direction classification and fold-level split conformal intervals.

The minutes are deliberately backdated to the meeting date. This is an **oracle information study**:
it estimates whether the information later documented in the minutes would have explained the
meeting-window price move if it had been perfectly available at the decision. It is not presented
as a deployable real-time strategy.

## Leakage boundary

The earlier monolithic design could reuse a price encoder selected using post-1993 observations in
early FOMC folds. This package does not. The fusion branch loads only a dedicated historical price
backbone whose **last training label is strictly earlier than the first aligned FOMC meeting**.
Checkpoint loading and the preflight audit both enforce this boundary.

The full-history daily price benchmark remains available as a separate experiment, but its weights
are never accepted by the FOMC fusion loader.

## Project map

```text
configs/                    Explicit model, extraction and paid-run switches
src/us10y_fomc/
  security/                 Plain-text .env loading and key validation
  data/                     FRED, Federal Reserve discovery/download and audits
  llm/                      14-feature schema, sentence IDs, grounding, cache, cost ledger
  features/                 Causal price cube, event alignment, training-only scalers
  models/                   CNN–Transformer, token encoders and cross-attention
  training/                 Backbones, ablations, conformal and walk-forward
  validation/               Leakage, masks, dimensions and frozen-weight gates
  evaluation/               Forecast metrics, paired tests and attention exports
  reporting/                Figures and evidence tables
scripts/                    Command-line entry points
notebooks/                  Thin preflight, master-run and post-process notebooks
tests/                      Deterministic contracts plus real-data integration checks
data/                       Local official inputs and extraction caches (Git-ignored)
outputs/                    Checkpoints, predictions, attention, metrics and reports
docs/                       Architecture, runbook and change log
```

## Quick start

Read [docs/RUNBOOK.md](docs/RUNBOOK.md) before enabling any paid switch.

```bash
conda activate torch_env
python -m pip install -e ".[dev]"
cp .env.example .env
python scripts/download_data.py
python scripts/run_preflight.py
```

Then validate model access, promote the already-approved pilot file, run the cached/resumable Luna
extraction, train the pre-FOMC backbone and run the four-model walk-forward study exactly in the
order described in the runbook.

## Four paired models

| Model | Forecasts | Uses prices | Uses rate facts | Uses aligned minutes |
|---|---|---:|---:|---:|
| `price_only` | direction + q05/q50/q95 | yes | no | no |
| `rate_only` | direction + q05/q50/q95 | yes | yes | no |
| `shuffled_text` | direction + q05/q50/q95 | yes | yes | no, alignment destroyed |
| `fusion` | direction + q05/q50/q95 | yes | yes | yes |

All four models are fitted and evaluated on the same fold meetings. The shuffled control permutes
the complete semantic row—including score, confidence and availability—inside each split while
leaving prices, rates and labels fixed.

## Monitoring

Long loops now show nested `tqdm` progress in both terminals and notebooks. Walk-forward training
shows fold, model, epoch, current validation loss, best loss and early-stopping state. Disable live
bars for CI or redirected logs with `US10Y_PROGRESS=0`.

Training progress is also appended durably to `outputs/runtime/*.jsonl`. From a second terminal:

```bash
tail -f outputs/runtime/historical_backbone.jsonl
tail -f outputs/runtime/walk_forward.jsonl
```

For a compact latest-record view with current epoch timing and stage ETA:

```bash
python scripts/monitor_runtime.py historical_backbone --watch
python scripts/monitor_runtime.py walk_forward --watch
```

Paid extraction progress is durable in the production cache, quarantine and cost ledger. Stopping
and restarting does not repay for already qualified pairs.

## v7.1 checkpoint rule

Every event model is selected by the minimum validation joint loss. The same loss drives patience;
balanced accuracy remains a diagnostic only. Training restores the selected weights, re-evaluates
their validation loss, and stores the selected epoch/loss plus the stopping contract in checkpoint
metadata. See [docs/V7_1_CHECKPOINT_FIX.md](docs/V7_1_CHECKPOINT_FIX.md).

## Validation

```bash
python -m compileall -q src scripts
US10Y_PROGRESS=0 pytest
```

The integration test refuses to invent synthetic market data: it runs only when the official local
panel, grounded event table and trained historical checkpoint are present; otherwise it reports a
skip. Pure contract tests still verify signatures, credentials, Fed ellipsis normalization and the
hard information-cutoff guard.

The bundled v7.1 release includes the real official data, grounded cache, historical backbone and
fresh eight-fold outputs used by the verification report. Runtime switches that retrain models or
make paid calls are off by default.
