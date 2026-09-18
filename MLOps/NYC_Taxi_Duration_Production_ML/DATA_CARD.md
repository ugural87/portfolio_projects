# Data card

## Source

NYC Taxi & Limousine Commission monthly yellow-taxi trip records in Parquet format. The application downloads
files from the source URL configured in `configs/base.yaml`; it does not redistribute them.

## Unit of observation

One completed yellow-taxi trip record. The training label is elapsed minutes from pickup to drop-off.

## Fields retained by the learning system

- `tpep_pickup_datetime`
- `tpep_dropoff_datetime` (label construction only)
- `PULocationID`
- `DOLocationID`
- `passenger_count`

All other source fields are discarded before model training.

## Known issues

Administrative trip records can contain invalid durations, missing passenger counts, invalid location codes,
exact duplicate rows, and behavior changes across vendors or collection versions. Zone IDs are coarse and do not encode
the driven route. Timestamps represent New York local operational time in the source.

## Quality controls

Required-column assertion, parse checks, source-month filter (stray rows from other months are removed and
counted as `out_of_period_rows`), configured value ranges including exclusion of zones 264 (Unknown) and
265 (Outside of NYC), exact-row duplicate removal over the five retained source fields, chronological
sorting, release gates for invalid/out-of-period/duplicate rates and valid volume, per-file SHA-256, and
time-range recording. The exact-row rule is deliberately conservative: without a stable source trip ID,
near-duplicates are retained rather than risking removal of legitimate trips with similar attributes.

## Leakage policy

Drop-off time is label-only. Realized distance, fare components, tip, total, rate code, payment type, and any
post-trip field are prohibited from features. The central feature transformer exposes an auditable allowlist.

## Governance

Raw files remain immutable and are excluded from Git. Production retention, access controls, encryption, and
privacy review belong to the deploying organization. Do not use this model for punitive individual decisions.
