# ADR 0005: Champion comparison, iteration selection, pinned runtime

Status: accepted (v1.1), comparison rule refined by ADR 0006 in v1.2

## Decisions

1. **Champion-challenger gate.** The retraining workflow extracts the model from the image tagged
   `production` and scores it on the candidate's test window. The candidate may not be worse than the
   champion by more than `gates.max_regression_vs_champion`. Absolute limits alone would allow a candidate
   that passes every threshold but is worse than what is serving. A champion that exists but cannot be
   evaluated blocks the release; no champion (first release) is a recorded skip.
2. **Iteration count on the chronological validation set.** Built-in early stopping holds out a random 10%
   of the training rows, which mixes periods inside a chronological design. The pipeline fits the maximum
   number of iterations, reads the validation MAE curve from `staged_predict`, and refits at the argmin.
   The test segment is not involved.
3. **One lock file for training, CI, the image and the champion.** A joblib artifact can only be trusted
   with the artifact-sensitive library versions that wrote it. The hash-locked runtime set is the single resolution; the
   loader also compares versions recorded in metadata and refuses mismatches. Changing the lock is a
   deliberate release that runs with `skip_champion`.

## Not adopted

Refitting on train+validation after gating would put the most recent days into the weights, at the cost of
shipping an artifact that is not the one evaluated. v1.1 ships the evaluated artifact.
