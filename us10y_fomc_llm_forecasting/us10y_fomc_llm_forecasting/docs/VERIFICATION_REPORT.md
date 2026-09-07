# v7.1 verification report

## Result

**PASS.** The complete v7.1 pipeline was executed against the supplied real v7 data/cache and
historical backbone. No synthetic market data and no new paid OpenAI request were used.

## Real-data execution

- 8/8 purged walk-forward folds completed.
- 4 paired event models per fold; 32/32 checkpoints created and loadable.
- 178 unique out-of-sample meeting targets produced, matching executed v7 exactly.
- Fold manifest, meeting dates, target dates, realised targets and classes are identical to v7.
- Full training runtime was approximately 9 minutes 23 seconds on the available CPU runtime.
- `tqdm` was observed at fold, model and epoch levels during the run.

## Checkpoint proof

For every one of the 32 training histories:

- exactly one epoch is marked as selected;
- its validation joint loss is the history minimum within the configured `1e-5` delta;
- validation joint loss also drives the 35-epoch patience counter;
- selected weights are restored and their validation loss is recomputed before serialization;
- checkpoint metadata records the metric, delta, patience, completed epochs, selected epoch, loss
  and diagnostic balanced accuracy;
- state-dict names and tensor shapes match the corresponding v7 checkpoint contract;
- every stored tensor is finite.

The machine-readable evidence is
`outputs/reports/v7_1_release_verification.json`. The verification can be repeated with:

```bash
US10Y_PROGRESS=0 python scripts/verify_v7_1_release.py \
  --baseline /path/to/executed/us10y_fomc_fusion
```

## Output contracts

- Prediction numeric fields contain no NaN or infinity.
- Conformal lower bounds are never above upper bounds.
- Pooled 90% interval coverage is 0.927 with mean width 72.16 bp.
- v7.1 MAE: fusion 13.331 bp; price-only 12.154 bp; rate-only 13.191 bp;
  shuffled-text 13.262 bp.
- v7 MAE for comparison: fusion 14.133 bp; price-only 12.209 bp; rate-only 12.890 bp;
  shuffled-text 14.121 bp.

These performance differences are expected because the selected epochs and therefore weights
changed. They are reported as experimental outcomes, not as a claim that fusion dominates the
controls; price-only has the lowest MAE in this run.

## Architecture and portability

- All Python sources under `src/us10y_fomc/models` and `src/us10y_fomc/features` hash-identically
  to executed v7.
- Legacy absolute macOS minutes paths safely rebase to the release's verified local corpus.
- FRED and Federal Reserve HTTP sessions advertise provider-appropriate content types.
- Terminal and notebook progress share the same `tqdm.auto` wrapper and can be disabled with
  `US10Y_PROGRESS=0`.

## Static and automated checks

- 13 pytest tests passed, including real-data batch and checkpoint-restoration tests.
- Python sources and all notebook code cells compile.
- TOML, JSON, JSONL, CSV, SVG and notebook files parse successfully.
- The official minutes files match the recorded raw-byte hashes.
- No populated `.env` or API credential is included.
