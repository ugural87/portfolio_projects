# Migration from the monolithic v6 notebook

The modular package retains the validated official-document, FRED-rate and Luna extraction logic,
but does not copy notebook state implicitly. Each stage now has explicit files and signatures.

The material architecture correction is the price-transfer boundary. In v6, the fitted price-only
encoder could have been selected using daily observations later than an early FOMC walk-forward
test fold. Freezing such weights prevents further updates but does not remove information already
embedded in them. The package therefore trains a separate price encoder with all labels ending
before the first FOMC meeting. That one immutable checkpoint is used across every event fold.

This is intentionally simpler than retraining a new price backbone in every fold. It sacrifices
post-1993 price-regime adaptation in exchange for a single transparent information boundary and
manageable runtime. The standalone full-history price benchmark remains available for the original
daily forecasting question and is explicitly tagged as ineligible for fusion.

Other preserved corrections include dynamic post-purge calibration allocation, a minimum of 19
calibration meetings, non-overlapping three-year test blocks, within-split shuffled-text controls,
rate-only controls, event-time DM tests, fold-block bootstrap comparisons and mask-aware attention
exports.
