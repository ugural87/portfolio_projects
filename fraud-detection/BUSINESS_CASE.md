# Business Case

## Decision

For each card transaction, estimate fraud probability and expected avoidable loss. Assign transactions to an hourly review queue subject to an explicit capacity limit.

## Cost model

`approve cost = P(fraud) × (amount + chargeback fee)`

`review cost = review cost + P(legitimate) × friction cost + P(fraud) × residual uncaptured loss`

The selected policy is chosen on the policy block by realized scenario cost and then frozen before final test evaluation.

## Reported outcomes

- alert count and rate;
- precision and fraud recall at capacity;
- fraud amount capture;
- estimated prevented fraud count and gross prevented loss;
- review and customer-friction costs;
- net savings, savings per review and savings per 1,000 transactions;
- stratified-bootstrap uncertainty;
- sensitivity to capture rate and false-positive friction.

## Claim boundary

The outputs are scenario estimates on public anonymized data. They must be described as estimated savings under stated assumptions. They are not realized savings from an identified bank.

