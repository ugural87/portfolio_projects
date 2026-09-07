# Tensor contracts

Let `B` be batch size, `T=60` the maximum intermeeting trading days, `M=7` Treasury
maturities, `C=4` causal price channels, `F=14` minutes features and `D=128` the model width.

| Tensor | Shape | Meaning |
|---|---:|---|
| price input | `B × C × T × M` | standardised level, 1-day change, 5-day change and 20-day realised volatility |
| price validity | `B × T` | true only for real intermeeting days |
| CNN daily tokens | `B × T × D` | local time/maturity patterns after maturity flattening |
| price summary | `B × D` | Transformer CLS representation |
| rate facts | `B × 2` | actual target change in bp and post-decision target midpoint |
| rate token | `B × 1 × D` | learned rate projection |
| semantic input | `B × F × 2` | training-standardised score and bounded confidence |
| semantic validity | `B × F` | true only when the feature is available and exactly grounded |
| semantic tokens | `B × F × D` | feature-specific projections with learned identity embeddings |
| full event sequence | `B × 15 × D` | one rate token followed by 14 semantic tokens |
| event→price attention | `B × 4 × 15 × T` | each event token routing to pre-meeting price days |
| price→event attention | `B × 4 × 1 × 15` | price summary routing to rate/text tokens |
| direction logits | `B × 3` | down, flat, up |
| ordered quantiles | `B × 3` | q05, q50, q95 in volatility-scaled target units |

The four attention heads have width `D / 4 = 32`. A scalar rate change cannot be numerically
"drowned" by a 128-dimensional text token: the rate pair and each semantic pair are separately
projected to the same width before attention. Scale parameters are fitted only on the current
fold's training meetings. Missing semantic tokens are excluded with a key-padding mask rather than
represented as ordinary zero-valued observations.
