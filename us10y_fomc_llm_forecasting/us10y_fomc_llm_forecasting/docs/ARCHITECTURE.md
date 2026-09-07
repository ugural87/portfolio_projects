# Architecture

The report-ready vector diagram is [`architecture.svg`](architecture.svg); its editable Mermaid
source is [`architecture.mmd`](architecture.mmd). It follows a layered C4-style view: external
systems, governed data/features, the preserved v7 neural model, and the walk-forward experiment.

## Model boundary

The price encoder used by every FOMC fold is the dedicated historical backbone. Its latest
training label must be earlier than the first aligned FOMC meeting. The separately trained modern
price-only benchmark is never loaded into the fusion model.

The model and feature-source files are byte-identical to executed v7:

- price path: 2D CNN branches, residual refinement and a three-layer price Transformer;
- event path: authoritative rate token plus 14 masked score/confidence semantic tokens;
- fusion: bidirectional four-head cross-attention and a residual fusion MLP;
- outputs: down/flat/up direction and ordered q05/q50/q95 forecasts.

## Paired experiment

Each fold fits four models on identical meetings:

1. `price_only`: historical price representation only;
2. `rate_only`: price representation plus the two-component rate token;
3. `shuffled_text`: full architecture with semantic rows permuted inside each split;
4. `fusion`: prices, rate facts and correctly aligned minutes features.

The fold contract is purged train → validation → calibration → test. Event checkpoints use minimum
validation joint loss for both best-weight selection and early-stopping patience. Selected weights
are restored and re-evaluated before split conformal calibration and sealed test inference.

Run `python scripts/build_architecture_html.py` only if the optional legacy interactive Plotly view
is also needed.
