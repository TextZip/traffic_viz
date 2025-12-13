# src/traffic_viz/config.py
from __future__ import annotations
from pathlib import Path
import os


def _repo_root() -> Path:
    # .../traffic_viz/src/traffic_viz/config.py → repo root
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    env = os.getenv("TRAFFIC_VIZ_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()

    repo_data = _repo_root() / "data"
    if repo_data.exists():
        return repo_data.resolve()

    return (Path.home() / "traffic_viz-data").resolve()


def cache_parquet_dir() -> Path:
    return get_data_dir() / "cache_parquet"


def meta_dir() -> Path:
    return get_data_dir() / "station_metadata"


def raw_5min_dir() -> Path:
    return get_data_dir() / "5min_data"
