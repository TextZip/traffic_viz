from __future__ import annotations
from src.travel_time import travel_time_samples
from src.metadata_tmdd import build_i5_nb_station_order
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "cache_parquet"
META_DIR = DATA / "station_metadata"


def main():
    # 1) Load cached parquet (already filtered to I-5 N ML)
    parquet_files = sorted(CACHE.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {CACHE}. Did you run preprocess_dir?")

    df = pd.concat([pd.read_parquet(p)
                   for p in parquet_files], ignore_index=True)

    print("Filtered rows total:", len(df))
    print("Unique stations:", df["Station"].nunique())
    print("Time range:", df["Timestamp"].min(), "->", df["Timestamp"].max())

    # 2) Load station metadata and build ordered station list
    meta = build_i5_nb_station_order([
        META_DIR / "d11_text_meta_2022_03_16.txt",
        META_DIR / "d12_text_meta_2023_12_05.txt",
        META_DIR / "d07_text_meta_2023_12_22.txt",
    ])

    ordered = meta["Station"].dropna().astype(int).tolist()
    print("Ordered I-5 NB ML stations:", len(ordered))

    # --- Refined endpoints: District-based SD -> LA ---
    # "First" station in District 11 (SD-ish)
    d11 = meta[meta["District"] == 11]
    d07 = meta[meta["District"] == 7]

    if len(d11) == 0 or len(d07) == 0:
        raise ValueError(
            "Could not find stations for District 11 and/or District 7 after filtering to I-5 N ML.")

    origin = int(d11["Station"].iloc[0])     # southern end (SD)
    dest = int(d07["Station"].iloc[-1])    # northern end (LA)

    print("Refined origin (D11):", origin, "Refined dest (D7):", dest)

    # 4) Travel time samples
    T = travel_time_samples(df, ordered, origin, dest, min_coverage=0.90)

    print("\nTravel time summary (minutes):")
    print(T.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
    print("Num samples:", len(T))


if __name__ == "__main__":
    main()
