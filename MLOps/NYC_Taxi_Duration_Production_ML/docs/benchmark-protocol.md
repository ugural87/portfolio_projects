# Real TLC benchmark and threshold-calibration protocol

Do not copy synthetic metrics into a production claim. Run this protocol against official immutable TLC
files and commit only aggregate reports, never raw trip data.

1. Resolve and download at least six consecutive published months; retain every SHA-256.
2. Train v1.2 on rolling months 1-3 and evaluate untouched months 4, 5, and 6 separately.
3. Record row quality, median baseline, validation/test MAE, p95 error, bias, route coverage, and breakdowns.
4. Repeat with the ordinal comparison arm and any challenger using the same windows and seeds.
5. Set release thresholds from the observed distribution plus an explicit business error budget. Do not tune
   against the final month and then call that month a holdout.
6. Save commands, resolved config hash, source checksums, runtime versions, and aggregate JSON under
   `reports/benchmarks/<run-date>/`. Review route/zone slices before approving thresholds.

Example:

```bash
taxi-ml train data/raw/yellow_tripdata_2025-01.parquet \
  data/raw/yellow_tripdata_2025-02.parquet \
  data/raw/yellow_tripdata_2025-03.parquet --model-version benchmark-2025q1
taxi-ml evaluate data/raw/yellow_tripdata_2025-04.parquet \
  --output-path reports/benchmarks/2025q1/eval-2025-04.json
```
