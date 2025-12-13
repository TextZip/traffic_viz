from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.metadata_tmdd import build_i5_nb_station_order
from src.travel_time import travel_time_samples


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "cache_parquet"
META_DIR = DATA / "station_metadata"


def mean_ci_normal(x: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """CLT-based CI for mean using Normal critical value."""
    x = x.dropna()
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    mu = x.mean()
    s = x.std(ddof=1)
    z = norm.ppf(1 - alpha/2)
    half = z * s / np.sqrt(n)
    return (mu - half, mu + half)


def add_time_buckets(dfT: pd.DataFrame, bin_minutes: int = 30) -> pd.DataFrame:
    """Adds weekday/weekend + time-of-day bins."""
    out = dfT.copy()
    out["weekday"] = out.index.weekday < 5
    minutes = out.index.hour * 60 + out.index.minute
    # 0..(1440/bin_minutes - 1)
    out["tod_bin"] = (minutes // bin_minutes).astype(int)
    out["tod_label"] = out["tod_bin"].apply(
        lambda b: f"{(b*bin_minutes)//60:02d}:{(b*bin_minutes)%60:02d}"
    )
    return out


def group_summary(dfT: pd.DataFrame, late_threshold_min: float = 180.0) -> pd.DataFrame:
    """
    Produces per-group summary:
      n, mean, std, quantiles, CI for mean, P(T>threshold), BTI
    """
    rows = []
    g = dfT.groupby(["weekday", "tod_bin"], sort=True)

    for (is_weekday, tod_bin), grp in g:
        x = grp["T"].dropna()
        n = len(x)
        if n == 0:
            continue

        mu = float(x.mean())
        sd = float(x.std(ddof=1)) if n > 1 else np.nan
        q50 = float(x.quantile(0.50))
        q90 = float(x.quantile(0.90))
        q95 = float(x.quantile(0.95))
        q99 = float(x.quantile(0.99))
        ci_lo, ci_hi = mean_ci_normal(x)
        p_late = float((x > late_threshold_min).mean())
        bti = float((q95 - mu) / mu) if mu > 0 else np.nan

        rows.append({
            "weekday": bool(is_weekday),
            "tod_bin": int(tod_bin),
            "start_time": grp["tod_label"].iloc[0],
            "n": n,
            "mean_min": mu,
            "std_min": sd,
            "p50_min": q50,
            "p90_min": q90,
            "p95_min": q95,
            "p99_min": q99,
            "ci95_lo_min": float(ci_lo),
            "ci95_hi_min": float(ci_hi),
            "p_late": p_late,
            "bti": bti,
        })

    return pd.DataFrame(rows).sort_values(["weekday", "tod_bin"]).reset_index(drop=True)


def main():
    # Load cached filtered 5-min data
    parquet_files = sorted(CACHE.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p)
                   for p in parquet_files], ignore_index=True)

    # Load & order metadata
    meta = build_i5_nb_station_order([
        META_DIR / "d11_text_meta_2022_03_16.txt",
        META_DIR / "d12_text_meta_2023_12_05.txt",
        META_DIR / "d07_text_meta_2023_12_22.txt",
    ])
    ordered = meta["Station"].dropna().astype(int).tolist()

    # Refined endpoints (District 11 -> District 7)
    d11 = meta[meta["District"] == 11]
    d07 = meta[meta["District"] == 7]
    origin = int(d11["Station"].iloc[0])
    dest = int(d07["Station"].iloc[-1])
    print("Origin (D11):", origin, "Dest (D7):", dest)

    # Compute travel time samples
    T = travel_time_samples(df, ordered, origin, dest, min_coverage=0.90)
    print("T samples:", len(T), "Time range:",
          T.index.min(), "->", T.index.max())

    # Build conditional dataframe
    dfT = T.to_frame("T")
    dfT = add_time_buckets(dfT, bin_minutes=30)

    # Summaries
    summary = group_summary(dfT, late_threshold_min=180.0)

    # Save for Streamlit later
    out_dir = DATA / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)

    dfT.reset_index().rename(columns={"index": "Timestamp"}).to_parquet(
        out_dir / "travel_time_samples.parquet", index=False)
    summary.to_csv(out_dir / "conditional_summary.csv", index=False)

    print("\nSaved:")
    print(" -", out_dir / "travel_time_samples.parquet")
    print(" -", out_dir / "conditional_summary.csv")

    # Print top 8 “worst” buckets by mean and by p95
    print("\nWorst by mean (top 8):")
    print(summary.sort_values("mean_min", ascending=False).head(8)[
        ["weekday", "start_time", "n", "mean_min", "p95_min", "p_late", "bti"]
    ])

    print("\nWorst by p95 (top 8):")
    print(summary.sort_values("p95_min", ascending=False).head(8)[
        ["weekday", "start_time", "n", "mean_min", "p95_min", "p_late", "bti"]
    ])


if __name__ == "__main__":
    main()
