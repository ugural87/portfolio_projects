# Customer Segmentation — RFM Clustering, Density Analysis and Dimensionality Reduction

Unsupervised segmentation of 4,338 customers built from 400k+ retail transactions
(UCI/Kaggle Online Retail, Dec 2010 – Dec 2011), carried through to an action per
segment — and, just as importantly, to an honest statement of **how much structure this
data actually contains.**

## Why this framing

Clustering projects usually end at "here are five segments" with no evidence that five
segments exist. This one treats that as the central question. Three independent methods
— EDA geometry, DBSCAN density analysis, and t-SNE neighborhood structure — converge on
the same verdict: the RFM space is **one continuous cloud with density gradients, not a
set of separated islands.** The segments are therefore presented as what they are —
*useful partitions of a continuum*, like deciles — while the density method's real
contribution turns out to be something else entirely: a principled key-account list.

The two headline numbers:

- **39% of customers generate 86% of revenue** (KMeans tier structure).
- **13.9% of customers flagged as DBSCAN "noise" hold 46.8% of revenue** — the accounts
  too extreme for any segment average to describe, identified without a single hand-set
  revenue threshold.

## Data

Kaggle: [`vijayuv/onlineretail`](https://www.kaggle.com/datasets/vijayuv/onlineretail)
(UCI Online Retail) — 541,909 transaction lines, one UK online retailer, Dec 2010 –
Dec 2011. Downloaded **automatically** on first run via `kagglehub`, with a mirrored
fallback; nothing goes in `data/` manually.

A transaction log is not a customer table, so cleaning rules are explicit and shared by
every notebook: drop unattributable guest checkouts (~25% missing `CustomerID`), drop
cancellation invoices and non-positive quantities/prices, then aggregate to customers as
of a snapshot date (last transaction + 1 day).

## Repository structure

```
.
├── notebooks/
│   ├── 01_eda_rfm.ipynb      # transaction-log audit, cleaning rules, RFM+ feature
│   │                         # construction, skew/log analysis, correlation structure
│   ├── 02_kmeans.ipynb       # elbow + silhouette curve + per-cluster silhouette
│   │                         # "knife" plots, segment profiling, action per segment
│   ├── 03_dbscan.ipynb       # min_samples/k-distance eps selection, eps grid,
│   │                         # knife plot, noise analysis, KMeans-vs-DBSCAN comparison
│   └── 04_pca_tsne.ipynb     # PCA variance/loadings/biplot, t-SNE with a perplexity
│                             # scan, both labelings overlaid on both projections
├── data/                     # auto-populated, git-ignored
├── requirements.txt
├── LICENSE
└── README.md
```

## Features

Six behavioral features per customer — the classic **RFM** plus three extensions that
give the dimensionality-reduction notebook something richer than a 3-axis space:

| Feature | Definition |
|---|---|
| `recency` | days since last purchase (snapshot-relative) |
| `frequency` | distinct invoices |
| `monetary` | total revenue |
| `tenure` | days since first purchase |
| `n_products` | distinct SKUs purchased |
| `avg_order_value` | monetary / frequency |

All are heavily right-skewed (raw skew 5–19), so every notebook shares one preprocessing
path: **`log1p` → `StandardScaler`**, which brings skews to ≈|1| and stops `monetary`'s
tail from dominating the Euclidean geometry that both k-means centroids and DBSCAN
ε-balls depend on.

## Key results

**KMeans (notebook 02).** The silhouette curve peaks at k=2 and decays smoothly — the
signature of partitioning a continuum rather than finding islands. The per-cluster knife
plots decide the trade: at **k=3** all three blades clear the average line with balanced
thicknesses (1,689 / 977 / 1,672 customers); from k=4 on, blades slip behind the line
and thin slivers appear.

| Segment | Customers | Revenue share | Median profile | Action |
|---|---|---|---|---|
| **Loyal high-value core** | 1,672 (39%) | **86%** | R 20d, F 6 orders, €2,092, 89 SKUs | protect: service, early access, churn-watch |
| **Recent one-timers** | 977 (23%) | 6% | R 32d, F 1, tenure 51d | convert: onboarding, second-purchase incentive |
| **Hibernating one-timers** | 1,689 (39%) | 8% | R 178d, F 1, tenure 263d | low-cost reactivation only |

**DBSCAN (notebook 03).** The eps grid maps both failure modes explicitly: at
eps ≥ 0.8 (including the elbow read of the k-distance graph, ~1.0) everything collapses
into a single cluster; at eps ≤ 0.5 it fragments into 10–13 micro-clusters with 40–75%
noise and *negative* silhouettes. Only eps = 0.7 gives a genuine two-cluster reading
(silhouette 0.274 excluding noise) — which reproduces the coarse engaged/disengaged
split at a slightly lower silhouette than KMeans achieves with three clusters. The noise
set is the finding: more recent, more frequent and higher-spending than clustered
customers, containing every whale in the book.

**PCA & t-SNE (notebook 04).** Loadings give the axes names — PC1 *customer value*
(53%), PC2 *lifecycle age* (tenure and recency loading together, 20%), PC3 *basket
style* (avg order value against frequency, 17%), PC4 *product breadth* (7%). Two
components carry 73% and three carry 90%. t-SNE at perplexity 5/30/50 shows one
connected body at every setting: no non-linear structure the linear view was hiding, and
the KMeans tiers occupy contiguous territory in all six maps.

## How to run

```bash
git clone https://github.com/ugural87/portfolio_projects.git
cd portfolio_projects/customer-segmentation
pip install -r requirements.txt
jupyter lab   # run notebooks in order; data downloads automatically on first run
```

Python ≥ 3.10. Notebook 04's t-SNE scan (three perplexities over 4,338 points) is the
slowest cell — expect a couple of minutes.

## Roadmap

- Segment **stability** analysis: bootstrap resampling and adjusted Rand index across
  seeds, to quantify how reproducible the tiers are
- Gaussian mixture models as a soft-assignment alternative — probabilities of belonging
  rather than hard tiers, which suits a continuum better
- Segment **migration matrix** across two half-year snapshots, turning the static
  segmentation into a lifecycle view
- CLV modelling (BG/NBD + Gamma-Gamma) on the same RFM base, for a forward-looking
  value estimate to sit beside the historical `monetary`

## License

MIT — see [LICENSE](LICENSE).
