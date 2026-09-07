# Final run checklist

## Before any paid call

- [ ] `.env` is plain text, excluded from Git and contains the intended project key.
- [ ] `run_paid_extraction = false` during the local-data and model-access preflight.
- [ ] Official-document integrity passes; any exclusion is limited to the known 2003-09-15 link.
- [ ] FRED rate anchors pass; minutes/FRED parser disagreements remain diagnostic.
- [ ] Pair token budget is present and the configured hard dollar budget is acceptable.
- [ ] The five-pair validated pilot is promoted under matching document and code signatures.
- [ ] Paid extraction and its confirmation switch are enabled together.

## Before training

- [ ] Every adjacent minutes pair is present in the qualified production cache.
- [ ] All 14 features have score, confidence, source availability, grounding and model-mask fields.
- [ ] Event alignment uses prices strictly after the previous meeting and strictly before the
  current meeting.
- [ ] Target dates occur after the current meeting; no target observation enters an input window.
- [ ] Historical-backbone checkpoint purpose is `fomc_historical_backbone`.
- [ ] Its last training label is strictly earlier than the first FOMC meeting.
- [ ] The modern standalone price benchmark checkpoint is not loaded by fusion.
- [ ] Fold scalers are fitted on fold training meetings only.
- [ ] Each post-purge calibration split contains at least 19 meetings.
- [ ] Projected walk-forward coverage exceeds 60% before the first model fit.

## After training

- [ ] Each of the 32 event-model histories marks exactly one selected checkpoint.
- [ ] The selected epoch is the minimum validation joint loss within `1e-5`.
- [ ] Restored validation loss matches the selected loss before the checkpoint is written.
- [ ] Checkpoint metadata records selection metric, delta, patience, selected epoch/loss and
  completed epochs.
- [ ] The four models have one prediction per identical OOS meeting.
- [ ] Pooled OOS meetings are unique and realised coverage equals the manifest projection.
- [ ] Ordered quantiles never cross.
- [ ] Padded price days and unavailable semantic features have zero exported attention.
- [ ] Full fusion and price-only forecasts are not mechanically identical.
- [ ] Fold-level conformal coverage, pooled coverage and interval width are reported.
- [ ] Paired DM tests use one step in meeting-event time (`hac_lag=0`).
- [ ] Fold-block bootstrap intervals accompany model comparisons.
- [ ] Attention is described as a routing diagnostic, not a causal explanation.
- [ ] The final twelve-month holdout remains sealed until the analysis is frozen.
