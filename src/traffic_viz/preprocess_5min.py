from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, Union, IO
from tqdm import tqdm

import pandas as pd


CORE_COLS = [
    "Timestamp", "Station", "District", "Freeway", "Direction", "LaneType",
    "StationLength", "Samples", "PctObserved", "TotalFlow", "AvgOccupancy", "AvgSpeed"
]


def _open_maybe_gzip(path: Path) -> IO[str]:
    """Open .gz as text, else open regular text."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def iter_filtered_chunks(path: Union[str, Path], chunksize: int = 250_000) -> Iterable[pd.DataFrame]:
    path = Path(path)
    usecols = list(range(12))  # first 12 columns only

    with _open_maybe_gzip(path) as f:
        reader = pd.read_csv(
            f,
            sep=",",
            header=None,
            usecols=usecols,
            names=CORE_COLS,
            chunksize=chunksize,
            engine="python",
        )

        for chunk in reader:
            # basic cleaning
            chunk["Timestamp"] = pd.to_datetime(
                chunk["Timestamp"], format="%m/%d/%Y %H:%M:%S", errors="coerce"
            )
            for c in ["Station", "District", "Freeway", "StationLength", "TotalFlow", "AvgOccupancy", "AvgSpeed"]:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

            chunk["Direction"] = chunk["Direction"].astype(
                str).str.strip().str.upper()
            chunk["LaneType"] = chunk["LaneType"].astype(
                str).str.strip().str.upper()

            # filter to I-5 NB mainline with valid speed
            chunk = chunk[
                (chunk["Freeway"] == 5) &
                (chunk["Direction"] == "N") &
                (chunk["LaneType"] == "ML") &
                (chunk["AvgSpeed"].notna()) &
                (chunk["AvgSpeed"] > 0) &
                (chunk["Timestamp"].notna())
            ]

            if len(chunk):
                yield chunk


def preprocess_dir(
    raw_dir: Union[str, Path],
    out_parquet: Union[str, Path],
    *,
    pattern: str = "d*_text_station_5min_*.txt.gz",
    chunksize: int = 250_000,
) -> None:
    """
    Convert raw PeMS station 5-min files -> per-file parquet cache, filtered to I-5 NB ML.

    raw_dir:
      directory containing raw files
    out_parquet:
      output directory to write .parquet files
    pattern:
      glob pattern inside raw_dir (supports .txt.gz or .txt depending on what you pass)
    """
    raw_dir = Path(raw_dir)
    out_parquet = Path(out_parquet)
    out_parquet.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matched pattern {pattern!r} in {raw_dir}")

    for fp in tqdm(files, desc="Preprocessing 5-min files", unit="file"):
        # Make a stable output name regardless of .txt.gz or .txt
        name = fp.name
        if name.endswith(".txt.gz"):
            stem = name[:-7]  # remove ".txt.gz"
        elif name.endswith(".txt"):
            stem = name[:-4]  # remove ".txt"
        else:
            stem = fp.stem

        out_fp = out_parquet / f"{stem}.parquet"

        parts = []
        for chunk in iter_filtered_chunks(fp, chunksize=chunksize):
            parts.append(chunk)

        if parts:
            df = pd.concat(parts, ignore_index=True)
            df.to_parquet(out_fp, index=False)
            print(f"[OK] {fp.name}: kept {len(df):,} rows -> {out_fp.name}")
        else:
            print(f"[WARN] {fp.name}: kept 0 rows")
