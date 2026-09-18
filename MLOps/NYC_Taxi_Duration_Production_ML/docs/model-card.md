# Model card: NYC taxi trip duration

## Intended use

Estimate trip duration in minutes at trip start for routing, customer information, or operational planning.

## Not intended for

Driver compensation, enforcement, individual performance ranking, emergency decisions, or use outside NYC
without retraining and validation.

## Inputs and output

Inputs: timezone-aware pickup timestamp, TLC pickup zone ID (1-263), intended TLC drop-off zone ID (1-263),
passenger count.
Output: conditional *median* duration in minutes, clipped to the validated 1-120 minute operating range.
It is not a mean: predictions should not be summed to plan aggregate capacity (see ADR 0003).

## Evaluation

Chronological train/validation/test split across a three-month rolling source window, after validating each
file against its own month. The final temporal tail remains validation/test data. Boosting iterations are
selected on the validation segment. Release gates cover data quality, MAE, p95
absolute error, improvement over a median baseline, validation-to-test stability, and no regression against
the production champion at the configured one-sided paired confidence level on the same test window.
`taxi-ml evaluate` scores a released model on later months;
error is broken down by hour, weekday and whether the route was seen in training. Metrics are written by every training run rather than
hard-coded in this card.

## Limitations

No live traffic, weather, event, road-closure, or route geometry signal is included. Zone IDs are coarse.
TLC records can contain reporting errors. The model can degrade under regime changes. Routes absent from
training fall back to zone-level and global statistics and have visibly higher error.
