from __future__ import annotations

import importlib
import platform
import sys


REQUIRED_MODULES = (
    "bs4",
    "matplotlib",
    "numpy",
    "openai",
    "pandas",
    "plotly",
    "pydantic",
    "pypdf",
    "requests",
    "scipy",
    "sklearn",
    "tiktoken",
    "torch",
    "tqdm",
)


if __name__ == "__main__":
    if not (3, 11) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError(f"Python 3.11 or 3.12 is required, found {platform.python_version()}.")
    missing = []
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            print(f"{name:12s} {getattr(module, '__version__', 'installed')}")
        except ImportError:
            missing.append(name)
    if missing:
        raise RuntimeError(f"Missing dependencies: {missing}. Install the project requirements.")
    print("Environment gate: PASS")
