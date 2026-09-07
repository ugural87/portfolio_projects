# v7.1 checkpoint fix

## Problem

Executed v7 tracked both validation balanced accuracy and validation joint loss, but checkpoint
selection could follow a different signal from the training objective. In 28 of 32 supplied v7
histories, the epoch with maximum balanced accuracy differed from the epoch with minimum validation
loss. That made early stopping and the saved regression/quantile model internally inconsistent.

## Resolution

v7.1 uses one explicit contract for all four event models:

1. compute validation joint loss after every epoch;
2. improve only when loss decreases by at least `1e-5`;
3. reset or increment the same metric's 35-epoch patience counter;
4. retain an in-memory copy of the best state dict;
5. restore that state when training stops;
6. run validation again and assert the restored loss matches the selected loss;
7. serialize weights and selection metadata together.

Balanced accuracy remains in histories, tqdm postfixes and metadata as a diagnostic; it never
selects weights. Quantile ordering and all v7 model/tensor contracts are unchanged.

## Observability

`tqdm.auto` progress is present in long data, extraction, training, bootstrap and notebook loops.
During event training the bar shows current validation loss, best loss, balanced accuracy and stale
epochs. Set `US10Y_PROGRESS=0` for CI, tests or redirected logs.

## Verification

`scripts/verify_v7_1_release.py` compares an executed v7 baseline with v7.1 and fails unless:

- folds and target identities match;
- protected model/feature source hashes match;
- all 32 histories select exactly one minimum-loss epoch;
- restored loss equals selected loss;
- checkpoint tensor names/shapes preserve the v7 contract;
- all predictions, intervals and stored tensors are finite and ordered.
