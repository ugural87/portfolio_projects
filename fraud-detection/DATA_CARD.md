# Data Card

## Source

The project uses OpenML dataset 1597, originally released by the Machine Learning Group at Université Libre de Bruxelles in collaboration with Worldline. The canonical raw ARFF file is downloaded directly because OpenML marks `Time` as a row identifier and higher-level loaders may omit it.

## Contents

- 284,807 card transactions observed over approximately two days in September 2013
- 492 fraud labels in the raw data
- `V1` to `V28`: PCA-anonymized numeric components
- `Time`: elapsed seconds from the first transaction
- `Amount`: transaction amount in unspecified currency units
- `Class`: fraud label

## Validation

The loader rejects the file unless all 31 expected columns, canonical row count, canonical fraud count, finite values, non-negative amounts and both target classes are present. The source SHA-256 digest is stored in `artifacts/data_audit.json`.

## Duplicate policy

Exact duplicate rows are removed before temporal partitioning. This prevents identical records from crossing model-development boundaries. Because the public data have no transaction identifier, exact duplicates cannot be proven to be duplicate business events; the removal is an explicit modeling choice and is reported in the audit artifact.

## Limitations

The window covers only about two days. Features are anonymized and contain no customer, merchant, device or network identifiers. There is no treatment log, post-decision loss outcome or label-availability timestamp. The data cannot establish production stability, fairness, causal intervention effects or realized bank savings.

