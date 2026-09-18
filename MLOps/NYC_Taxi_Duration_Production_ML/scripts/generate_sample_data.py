"""Deterministic synthetic TLC-shaped data for tests, CI and the quick start.

Zones get random latent coordinates. Duration depends on the distance between those
coordinates, not on the difference of zone IDs: real TLC LocationIDs follow the alphabetical
order of zone names, so an ID-difference signal would flatter an ordinal encoding that has no
geographic meaning on real data. Time effects are multiplicative, as they are for traffic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_ZONES = 263


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/raw/sample.parquet"))
    args = parser.parse_args()

    geography = np.random.default_rng(0)  # fixed across files: zones do not move
    coords = geography.uniform(0.0, 12.0, size=(N_ZONES + 1, 2))  # km, index = zone id

    rng = np.random.default_rng(args.seed)
    pickup = pd.date_range(args.start, periods=args.rows, freq="1min")
    popularity = geography.dirichlet(np.full(N_ZONES, 0.1))  # few busy zones, long tail
    pu = rng.choice(np.arange(1, N_ZONES + 1), size=args.rows, p=popularity)
    do = rng.choice(np.arange(1, N_ZONES + 1), size=args.rows, p=popularity)
    passenger = rng.integers(1, 7, args.rows)

    distance_km = np.linalg.norm(coords[pu] - coords[do], axis=1) * 1.3  # street detour factor
    hour = pickup.hour.to_numpy()
    rush = ((hour >= 7) & (hour <= 10)) | ((hour >= 16) & (hour <= 20))
    weekend = pickup.dayofweek.to_numpy() >= 5
    speed_kmh = 22.0 * np.where(rush, 0.7, 1.0) * np.where(weekend, 1.15, 1.0)
    duration = (3.0 + 60.0 * distance_km / speed_kmh) * rng.lognormal(0.0, 0.25, args.rows)
    duration = np.clip(duration, 1.1, 115.0)

    frame = pd.DataFrame(
        {
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": pickup + pd.to_timedelta(duration, unit="m"),
            "PULocationID": pu,
            "DOLocationID": do,
            "passenger_count": passenger,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
