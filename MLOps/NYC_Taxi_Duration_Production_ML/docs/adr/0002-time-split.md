# ADR 0002: Chronological evaluation

Status: accepted

## Decision

Sort records by pickup time, train on the earliest 70%, validate on the next 15%, and reserve the latest
15% as a final test segment. Random splitting is prohibited for release evaluation.

## Rationale

Production predicts future trips. A chronological split exposes temporal drift and avoids optimistic
mixing of near-identical time regimes across train and test.

