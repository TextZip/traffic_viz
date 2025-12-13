from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import pandas as pd


def read_tmdd_meta(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    # sniff delimiter
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(50_000)
    dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", "|", ";"])
    sep = dialect.delimiter

    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip() for c in df.columns]

    # Normalize lookup helper
    lower_map = {c.lower().replace(" ", "").replace("_", "")                 : c for c in df.columns}

    def pick(*candidates: str):
        for cand in candidates:
            key = cand.lower().replace(" ", "").replace("_", "")
            if key in lower_map:
                return lower_map[key]
        return None

    # Canonical cols
    col_id = pick("ID")
    col_fwy = pick("Fwy", "Freeway", "FreewayNumber", "Freeway#")
    col_dir = pick("Dir", "Direction")
    col_dist = pick("District")
    col_type = pick("Type", "StationType")
    col_lat = pick("Latitude", "Lat")
    col_lon = pick("Longitude", "Lon")
    col_len = pick("Length", "StationLength")
    col_name = pick("Name")

    # IMPORTANT: keep Abs_PM and State_PM separate
    col_state_pm = pick("State_PM", "StatePM", "State PM",
                        "StatePostmile", "StatePostMile")
    col_abs_pm = pick("Abs_PM", "AbsPM", "Abs PM",
                      "AbsolutePM", "AbsolutePostmile", "AbsPostmile")

    # Rename to stable internal names (matching your metadata files)
    rename = {}
    if col_id:
        rename[col_id] = "Station"
    if col_fwy:
        rename[col_fwy] = "Freeway"
    if col_dir:
        rename[col_dir] = "Direction"
    if col_dist:
        rename[col_dist] = "District"
    if col_type:
        rename[col_type] = "LaneType"      # ML/OR/FR/HV/...
    if col_lat:
        rename[col_lat] = "Latitude"
    if col_lon:
        rename[col_lon] = "Longitude"
    if col_len:
        rename[col_len] = "Length"
    if col_name:
        rename[col_name] = "Name"
    if col_state_pm:
        rename[col_state_pm] = "State_PM"
    if col_abs_pm:
        rename[col_abs_pm] = "Abs_PM"

    df = df.rename(columns=rename)

    # Numeric coercion
    for c in ["Station", "Freeway", "District", "Latitude", "Longitude", "Length", "State_PM", "Abs_PM"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clean strings
    if "Direction" in df.columns:
        df["Direction"] = df["Direction"].astype(str).str.strip().str.upper()
    if "LaneType" in df.columns:
        df["LaneType"] = df["LaneType"].astype(str).str.strip().str.upper()
    if "Name" in df.columns:
        df["Name"] = df["Name"].astype(str).str.strip()

    return df


def _pca_order_latlon(m: pd.DataFrame) -> pd.Series:
    """Fallback ordering if Abs_PM unavailable: PCA projection on (lon, lat)."""
    X = np.vstack([m["Longitude"].to_numpy(float),
                  m["Latitude"].to_numpy(float)]).T
    X = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    pc1 = vt[0]
    proj = X @ pc1
    return pd.Series(proj, index=m.index)


def build_i5_nb_station_order(meta_paths: list[str | Path]) -> pd.DataFrame:
    meta = pd.concat([read_tmdd_meta(p)
                     for p in meta_paths], ignore_index=True)

    required = ["Station", "Freeway", "Direction", "District"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise KeyError(
            f"Metadata missing required columns: {missing}. Found: {list(meta.columns)}")

    # HARD FILTER: I-5 Northbound only (and mainline if present)
    m = meta[(meta["Freeway"] == 5) & (meta["Direction"] == "N")].copy()

    if "LaneType" in m.columns:
        m = m[m["LaneType"] == "ML"]

    # Keep only usable rows
    m = m.dropna(subset=["Station", "Latitude", "Longitude"]).copy()

    # De-dup by station id (keep last)
    m = m.drop_duplicates(subset=["Station"], keep="last").copy()

    # ORDERING: Abs_PM preferred (global). If missing, State_PM (local-ish), else PCA.
    if "Abs_PM" in m.columns and m["Abs_PM"].notna().sum() > 50:
        m = m.dropna(subset=["Abs_PM"]).sort_values("Abs_PM").copy()
    elif "State_PM" in m.columns and m["State_PM"].notna().sum() > 50:
        # This can reset across counties; still better than pure Lat, but not perfect.
        m = m.dropna(subset=["State_PM"]).sort_values(
            ["District", "State_PM"]).copy()
    else:
        # Last resort: PCA along corridor direction.
        m["_pca"] = _pca_order_latlon(m)
        m = m.sort_values("_pca").drop(columns=["_pca"]).copy()

    # Make Station int for downstream
    m["Station"] = m["Station"].astype(int)
    return m
