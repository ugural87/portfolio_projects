from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import os
import re
import tempfile
import urllib.request

import numpy as np
import pandas as pd


OPENML_RAW_URL = "https://www.openml.org/data/v1/download/1673544/creditcard.arff"
EXPECTED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
EXPECTED_ROWS = 284_807
EXPECTED_FRAUDS = 492


@dataclass(frozen=True)
class DataAudit:
    raw_rows: int
    modeling_rows: int
    columns: int
    frauds_raw: int
    frauds_modeling: int
    fraud_rate_modeling: float
    exact_duplicates_removed: int
    missing_values: int
    time_min_seconds: float
    time_max_seconds: float
    source_sha256: str


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frame(frame: pd.DataFrame, canonical_counts: bool = True) -> None:
    missing = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(EXPECTED_COLUMNS))
    if missing or extra:
        raise ValueError(f"Unexpected schema. Missing={missing}; extra={extra}")
    if canonical_counts and len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows, found {len(frame):,}")
    frame["Class"] = pd.to_numeric(frame["Class"], errors="raise").astype("int8")
    if canonical_counts and int(frame["Class"].sum()) != EXPECTED_FRAUDS:
        raise ValueError(f"Expected {EXPECTED_FRAUDS} frauds, found {int(frame['Class'].sum())}")
    if set(frame["Class"].unique()) != {0, 1}:
        raise ValueError("Class must contain exactly {0, 1}")
    if int(frame.isna().sum().sum()) != 0:
        raise ValueError("Dataset contains missing values")
    if not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Dataset contains non-finite numeric values")
    if (frame["Amount"] < 0).any():
        raise ValueError("Amount contains negative values")
    if (frame["Time"] < 0).any():
        raise ValueError("Time contains negative values")


def _arff_columns(path: Path) -> list[str]:
    columns: list[str] = []
    pattern = re.compile(r"^@attribute\s+['\"]?([^'\"\s]+)['\"]?\s+", re.IGNORECASE)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.lower() == "@data":
                break
            match = pattern.match(stripped)
            if match:
                columns.append(match.group(1))
    return columns


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "fraud-detection-research/1.0"})
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(request, timeout=120) as response:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                tmp.write(block)
    if tmp_path.stat().st_size < 100_000_000:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Downloaded file is unexpectedly small")
    os.replace(tmp_path, destination)


def _arff_to_csv(arff_path: Path, csv_path: Path) -> None:
    columns = _arff_columns(arff_path)
    if columns != EXPECTED_COLUMNS:
        raise ValueError(f"Raw OpenML schema mismatch: {columns}")
    frame = pd.read_csv(
        arff_path,
        comment="@",
        header=None,
        names=columns,
        quotechar="'",
        skip_blank_lines=True,
        low_memory=False,
    )
    validate_frame(frame, canonical_counts=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=csv_path.parent, suffix=".csv") as tmp:
        tmp_path = Path(tmp.name)
    frame.to_csv(tmp_path, index=False)
    os.replace(tmp_path, csv_path)


def ensure_dataset(data_dir: str | Path, force: bool = False) -> tuple[Path, Path]:
    data_dir = Path(data_dir)
    csv_path = data_dir / "creditcard.csv"
    arff_path = data_dir / "creditcard.arff"
    if csv_path.exists() and not force:
        try:
            cached = pd.read_csv(csv_path)
            validate_frame(cached, canonical_counts=True)
            return csv_path, arff_path
        except Exception:
            csv_path.unlink(missing_ok=True)
    if not arff_path.exists() or force:
        _download(OPENML_RAW_URL, arff_path)
    _arff_to_csv(arff_path, csv_path)
    return csv_path, arff_path


def load_dataset(
    data_dir: str | Path,
    drop_exact_duplicates: bool = True,
    force_download: bool = False,
) -> tuple[pd.DataFrame, DataAudit]:
    csv_path, arff_path = ensure_dataset(data_dir, force=force_download)
    raw = pd.read_csv(csv_path)
    validate_frame(raw, canonical_counts=True)
    duplicates = int(raw.duplicated().sum())
    frame = raw.drop_duplicates().copy() if drop_exact_duplicates else raw.copy()
    frame = frame.sort_values("Time", kind="mergesort").reset_index(drop=True)
    audit = DataAudit(
        raw_rows=len(raw),
        modeling_rows=len(frame),
        columns=frame.shape[1],
        frauds_raw=int(raw["Class"].sum()),
        frauds_modeling=int(frame["Class"].sum()),
        fraud_rate_modeling=float(frame["Class"].mean()),
        exact_duplicates_removed=duplicates if drop_exact_duplicates else 0,
        missing_values=int(frame.isna().sum().sum()),
        time_min_seconds=float(frame["Time"].min()),
        time_max_seconds=float(frame["Time"].max()),
        source_sha256=file_sha256(arff_path),
    )
    return frame, audit


def audit_to_dict(audit: DataAudit) -> dict:
    return asdict(audit)
