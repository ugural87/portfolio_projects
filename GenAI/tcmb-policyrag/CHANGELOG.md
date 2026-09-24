# Changelog

## 1.0.5 — 2026-09-24

- Added native build-index progress reporting with completed/total chunks,
  percentage and cache-hit counts.
- Kept query-time embeddings quiet while enabling progress only for corpus builds.

## 1.0.4 — 2026-09-24

- Applied the CI NumPy type-check constraint to documented local development
  installs and `make install`, keeping local lint behavior aligned with CI.

## 1.0.3 — 2026-09-24

- Parameterized the merge regression test so newest-wins behavior is verified
  independently of alphabetical shard order.
- Removed built wheels from the source repository and ignored standard build outputs.
- Moved the NumPy stub compatibility cap from runtime dependencies to a CI-only
  type-check constraint.

## 1.0.2 — 2026-09-24

- Fixed plain `pytest` collection for repo-local crawler tests.
- Made shard merges prefer the newest fetched document and matching raw HTML.
- Added merge conflict accounting and regression coverage.
- Included every source shard required to reproduce the committed corpus snapshot.
- Restored strict NumPy type checking by constraining incompatible future stubs.
- Added Python 3.11/3.12 CI coverage, package builds, Docker builds and tag releases.

## 1.0.1 — 2026-09-24

- Added deterministic shard merging and crawler-compatible aggregate reports.
- Included non-numeric TCMB announcement slugs; retained PDF-only open letters and
  presentations as explicitly excluded archive metadata.
- Corrected Turkish policy-instrument suffix matching.
- Added validated date filters to the CLI and Streamlit UI.
- Made citations deterministic and removed unknown citation markers from answers.
- Prevented query embeddings from entering the persistent corpus cache.
- Hardened token-window decoding against replacement characters.
- Raised the minimum OpenAI SDK version for Responses API compatibility.
- Added strict mypy and Ruff formatting checks to CI.
