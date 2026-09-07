from __future__ import annotations

from _bootstrap import PROJECT_ROOT

from us10y_fomc.config import ProjectPaths, load_project_config
from us10y_fomc.data.treasury_data import load_treasury_panel
from us10y_fomc.features.price_features import build_price_feature_bundle
from us10y_fomc.training.price_benchmark import train_price_benchmark


if __name__ == "__main__":
    config = load_project_config(PROJECT_ROOT)
    if not config.runtime.train_price_benchmark:
        raise RuntimeError("Set train_price_benchmark=true in configs/runtime.toml.")
    paths = ProjectPaths(PROJECT_ROOT).ensure()
    panel = load_treasury_panel(paths.data / "raw")
    bundle = build_price_feature_bundle(panel, config.price)
    print(
        train_price_benchmark(
            bundle,
            config.price,
            paths.outputs / "checkpoints" / "price" / "price_benchmark.pt",
            paths.outputs / "predictions" / "price_benchmark_predictions.csv",
            paths.outputs / "runtime" / "price_benchmark.jsonl",
        )
    )
