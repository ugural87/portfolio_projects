from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def progress_enabled() -> bool:
    """Return whether user-facing progress bars are enabled for this process."""
    value = os.getenv("US10Y_PROGRESS", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "item",
    leave: bool = True,
) -> tqdm[Iterator[T]]:
    """Create one consistently configured terminal/Jupyter progress bar."""
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        mininterval=0.25,
        disable=not progress_enabled(),
    )
