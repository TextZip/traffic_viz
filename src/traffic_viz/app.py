import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import requests
from functools import lru_cache

from traffic_viz.metadata_tmdd import build_i5_nb_station_order
from traffic_viz.travel_time import travel_time_samples
from traffic_viz.config import cache_parquet_dir, meta_dir


# ROOT = Path(__file__).resolve().parent
# DATA = ROOT / "data"

CACHE = cache_parquet_dir()
if not CACHE.exists() or not any(CACHE.glob("*.parquet")):
    import streamlit as st
    st.error(
        "No parquet cache found.\n\n"
        "Run preprocessing first:\n"
        "  traffic_viz preprocess"
    )
    st.stop()

META_DIR = meta_dir()

# CACHE = DATA / "cache_parquet"
# META_DIR = DATA / "station_metadata"
# (lon_min, lat_max, lon_max, lat_min)
LA_SD_VIEWBOX = (-118.9, 34.6, -116.8, 32.4)

st.set_page_config(page_title="I-5 Travel Time Predictor", layout="wide")
st.markdown(
    """
    <style>
    /* ---- Metric value (big number) ---- */
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 600;
    }

    /* ---- Metric label ---- */
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        opacity: 0.85;
    }

    /* ---- Metric delta (if used) ---- */
    div[data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("I-5 NB Travel Time Predictor (PeMS)")
st.caption(
    "Pick origin/destination stations by name and get probabilistic travel-time stats.")

# ---------------------- Data loaders ----------------------


@st.cache_data
def load_filtered_5min():
    parquet_files = sorted(CACHE.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {CACHE}.")
    return pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)


@st.cache_data
def load_meta_and_order():
    meta = build_i5_nb_station_order([
        META_DIR / "d11_text_meta_2022_03_16.txt",
        META_DIR / "d12_text_meta_2023_12_05.txt",
        META_DIR / "d07_text_meta_2023_12_22.txt",
    ])
    ordered = meta["Station"].dropna().astype(int).tolist()
    return meta, ordered


@st.cache_data(show_spinner=True)
def compute_T_for_corridor(df, ordered, origin: int, dest: int, min_coverage: float):
    T = travel_time_samples(df, ordered, origin, dest,
                            min_coverage=min_coverage)
    out = T.to_frame("T")
    out["weekday"] = out.index.weekday < 5
    minutes = out.index.hour * 60 + out.index.minute
    out["tod_bin"] = (minutes // 30).astype(int)
    out["tod_label"] = out["tod_bin"].apply(
        lambda b: f"{(b*30)//60:02d}:{(b*30)%60:02d}")
    return out


@st.cache_data(show_spinner=False)
def station_bucket_speed(df: pd.DataFrame, stations: list[int], weekday: bool, time_str: str) -> pd.DataFrame:
    """
    Returns per-station mean speed for the selected (weekday/weekend, 30-min bin).
    df is your filtered 5-min parquet cache with columns at least:
      Station, Timestamp, AvgSpeed
    """
    # Parse time_str like "16:30"
    hh = int(time_str[:2])
    mm = int(time_str[3:])
    target_bin = (hh * 60 + mm) // 30

    d = df[df["Station"].isin(stations)].copy()
    d = d.dropna(subset=["Timestamp", "AvgSpeed"])
    d["weekday"] = d["Timestamp"].dt.weekday < 5
    minutes = d["Timestamp"].dt.hour * 60 + d["Timestamp"].dt.minute
    d["tod_bin"] = (minutes // 30).astype(int)

    d = d[(d["weekday"] == weekday) & (d["tod_bin"] == target_bin)]
    if d.empty:
        return pd.DataFrame({"Station": stations, "mean_speed": np.nan})

    out = (
        d.groupby("Station", as_index=False)
         .agg(mean_speed=("AvgSpeed", "mean"), n=("AvgSpeed", "size"))
    )
    return out


df = load_filtered_5min()
meta, ordered = load_meta_and_order()


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def corridor_viewbox_ca(meta_ml: pd.DataFrame, pad_deg: float = 0.15):
    """
    Bounding box tightly enclosing the I-5 NB ML stations.
    pad_deg ≈ 0.15° ≈ 10–15 km buffer.

    Returns (lon_min, lat_max, lon_max, lat_min) for Nominatim.
    """
    lat_min = float(meta_ml["Latitude"].min()) - pad_deg
    lat_max = float(meta_ml["Latitude"].max()) + pad_deg
    lon_min = float(meta_ml["Longitude"].min()) - pad_deg
    lon_max = float(meta_ml["Longitude"].max()) + pad_deg

    return lon_min, lat_max, lon_max, lat_min


@st.cache_data(show_spinner=False)
def geocode_us_bounded_candidates(query: str, viewbox, limit: int = 5):
    q = (query or "").strip()
    if not q:
        return []

    lon_min, lat_max, lon_max, lat_min = viewbox

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": q,
        "format": "json",
        "limit": int(limit),
        "countrycodes": "us",
        "viewbox": f"{lon_min},{lat_max},{lon_max},{lat_min}",
        "bounded": 1,  # must be inside viewbox
        "addressdetails": 1,
    }
    headers = {"User-Agent": "i5-travel-time-predictor/1.0 (streamlit)"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return [
            {"display_name": it.get("display_name", ""), "lat": float(
                it["lat"]), "lon": float(it["lon"])}
            for it in data
        ]
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def geocode_us_candidates_with_fallback(query: str, viewbox, limit: int = 5):
    # strict
    out = geocode_us_bounded_candidates(query, viewbox=viewbox, limit=limit)
    if out:
        return out

    # fallback: biased but not forced inside viewbox
    lon_min, lat_max, lon_max, lat_min = viewbox
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "limit": int(limit),
        "countrycodes": "us",
        "viewbox": f"{lon_min},{lat_max},{lon_max},{lat_min}",
        "bounded": 0,  # allow outside, but bias ranking
        "addressdetails": 1,
    }
    headers = {"User-Agent": "i5-travel-time-predictor/1.0 (streamlit)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return [
            {"display_name": it.get("display_name", ""), "lat": float(
                it["lat"]), "lon": float(it["lon"])}
            for it in data
        ]
    except Exception:
        return []


def nearest_station_ml(meta_ml: pd.DataFrame, lat: float, lon: float):
    """
    Returns (station_id, distance_km, matched_row)
    meta_ml must have Station, Latitude, Longitude
    """
    lats = meta_ml["Latitude"].to_numpy(dtype=float)
    lons = meta_ml["Longitude"].to_numpy(dtype=float)

    # vectorized haversine (km) using your haversine_km()
    d = haversine_km(lat, lon, lats, lons)
    idx = int(np.argmin(d))
    row = meta_ml.iloc[idx]
    return int(row["Station"]), float(d[idx]), row
# ---------------------- Build station labels ----------------------
# meta_disp = meta.copy()


def normalize_meta(meta: pd.DataFrame) -> pd.DataFrame:
    m = meta.copy()

    # Rename PeMS metadata columns -> our canonical names
    rename = {
        "ID": "Station",
        "Fwy": "Freeway",
        "Dir": "Direction",
        "Type": "LaneType",   # metadata uses Type (ML/OR/FR/HV/CD/FF)
    }
    for k, v in rename.items():
        if k in m.columns and v not in m.columns:
            m = m.rename(columns={k: v})

    # Ensure lat/lon column names match what we use later
    # (your metadata already has Latitude/Longitude; keep as-is)
    if "Lat" in m.columns and "Latitude" not in m.columns:
        m = m.rename(columns={"Lat": "Latitude"})
    if "Lon" in m.columns and "Longitude" not in m.columns:
        m = m.rename(columns={"Lon": "Longitude"})

    # Types
    if "Station" in m.columns:
        m["Station"] = pd.to_numeric(m["Station"], errors="coerce")
    if "Latitude" in m.columns:
        m["Latitude"] = pd.to_numeric(m["Latitude"], errors="coerce")
    if "Longitude" in m.columns:
        m["Longitude"] = pd.to_numeric(m["Longitude"], errors="coerce")

    # String cleanup
    for c in ["Name", "Direction", "LaneType"]:
        if c in m.columns:
            m[c] = m[c].astype(str).str.strip()

    return m


meta_disp = normalize_meta(meta)

# Defensive: ensure Name exists
if "Name" not in meta_disp.columns:
    meta_disp["Name"] = "(no name)"

# Ensure Station is int-like for dropdown mapping
meta_disp = meta_disp.dropna(subset=["Station"]).copy()
meta_disp["Station"] = pd.to_numeric(meta_disp["Station"], errors="coerce")
meta_disp = meta_disp.dropna(subset=["Station"]).copy()
meta_disp["Station"] = meta_disp["Station"].astype(int)

# Coerce PM fields if present
if "Abs_PM" in meta_disp.columns:
    meta_disp["Abs_PM"] = pd.to_numeric(meta_disp["Abs_PM"], errors="coerce")
else:
    meta_disp["Abs_PM"] = np.nan

if "State_PM" in meta_disp.columns:
    meta_disp["State_PM"] = pd.to_numeric(
        meta_disp["State_PM"], errors="coerce")
else:
    meta_disp["State_PM"] = np.nan

# Build label: show BOTH, clearly named
meta_disp["label"] = (
    meta_disp["Name"].astype(str).fillna("(no name)") +
    " | ID=" + meta_disp["Station"].astype(str) +
    " | D=" + meta_disp["District"].astype(int).astype(str) +
    " | AbsPM=" + meta_disp["Abs_PM"].round(2).astype(str) +
    " | StatePM=" + meta_disp["State_PM"].round(2).astype(str)
)

# Keep only stations in ordering
meta_disp = meta_disp[meta_disp["Station"].isin(ordered)].copy()

# Sort dropdown by Abs_PM if available; else by Latitude
if meta_disp["Abs_PM"].notna().any():
    meta_disp = meta_disp.sort_values("Abs_PM")
elif "Latitude" in meta_disp.columns and meta_disp["Latitude"].notna().any():
    meta_disp = meta_disp.sort_values("Latitude")


def filter_mainline_nb_i5(meta_disp: pd.DataFrame) -> pd.DataFrame:
    m = meta_disp.copy()

    # Require coords for nearest-station search
    m = m.dropna(subset=["Station", "Latitude", "Longitude"]).copy()

    if "Freeway" in m.columns:
        m = m[pd.to_numeric(m["Freeway"], errors="coerce") == 5]
    if "Direction" in m.columns:
        m = m[m["Direction"].astype(str).str.upper().str.strip() == "N"]
    if "LaneType" in m.columns:
        m = m[m["LaneType"].astype(str).str.upper().str.strip() == "ML"]

    # clean station type
    m["Station"] = pd.to_numeric(m["Station"], errors="coerce")
    m = m.dropna(subset=["Station"]).copy()
    m["Station"] = m["Station"].astype(int)

    # Keep ordering stations only (belt+suspenders)
    m = m[m["Station"].isin(ordered)].copy()

    return m


meta_ml = filter_mainline_nb_i5(meta_disp)

# Build dropdown labels ONLY from ML stations
labels_ml = meta_ml["label"].tolist()
label_to_station_ml = dict(zip(meta_ml["label"], meta_ml["Station"]))
station_to_label_ml = dict(zip(meta_ml["Station"], meta_ml["label"]))


labels = meta_disp["label"].tolist()
label_to_station = dict(zip(meta_disp["label"], meta_disp["Station"]))

# # Defensive: if Name missing
# if "Name" not in meta_disp.columns:
#     meta_disp["Name"] = "(no name)"

# # Ensure Station is int-like for dropdown mapping
# meta_disp = meta_disp.dropna(subset=["Station"]).copy()
# meta_disp["Station"] = meta_disp["Station"].astype(int)

# # when building meta_disp["label"]
# spm = meta_disp["State_PM"] if "State_PM" in meta_disp.columns else np.nan
# apm = meta_disp["Abs_PM"] if "Abs_PM" in meta_disp.columns else np.nan

# meta_disp["label"] = (
#     meta_disp["Name"].astype(str).fillna("(no name)") +
#     " | ID=" + meta_disp["Station"].astype(str) +
#     " | D=" + meta_disp["District"].astype(int).astype(str) +
#     ((" | AbsPM=" + apm.round(2).astype(str)) if "Abs_PM" in meta_disp.columns else "") +
#     ((" | StatePM=" + spm.round(2).astype(str))
#      if "State_PM" in meta_disp.columns else "")
# )


# meta_disp = meta_disp[meta_disp["Station"].isin(ordered)].copy()

# sort_key = "Postmile" if "Postmile" in meta_disp.columns else (
#     "Latitude" if "Latitude" in meta_disp.columns else None)

# # sort_key = "Postmile" if "Postmile" in meta_disp.columns else (
# #     "Lat" if "Lat" in meta_disp.columns else None)
# if sort_key and sort_key in meta_disp.columns:
#     meta_disp = meta_disp.sort_values(sort_key)

# labels = meta_disp["label"].tolist()
# label_to_station = dict(zip(meta_disp["label"], meta_disp["Station"]))

default_origin = int(meta_disp[meta_disp["District"] == 11]["Station"].iloc[0])
default_dest = int(meta_disp[meta_disp["District"] == 7]["Station"].iloc[-1])
origin_label_default = meta_disp[meta_disp["Station"]
                                 == default_origin]["label"].iloc[0]
dest_label_default = meta_disp[meta_disp["Station"]
                               == default_dest]["label"].iloc[0]


def idx_of_station(st_id: int) -> int:
    return ordered.index(st_id)


# ---------------------- Sidebar controls ----------------------
st.sidebar.header("Controls")

# --- pick defaults, but ensure they exist in ML list ---
default_origin = int(meta_ml[meta_ml["District"] == 11]["Station"].iloc[5])
default_dest = int(meta_ml[meta_ml["District"] == 7]["Station"].iloc[-5])
origin_label_default = station_to_label_ml[default_origin]
dest_label_default = station_to_label_ml[default_dest]

# Init state
if "origin_label" not in st.session_state:
    st.session_state.origin_label = origin_label_default
if "dest_label" not in st.session_state:
    st.session_state.dest_label = dest_label_default

# st.sidebar.subheader("Station selection")

with st.sidebar.expander("Search by location", expanded=False):
    start_q = st.text_input("Start location", "", key="start_location_q")
    end_q = st.text_input("End location", "", key="end_location_q")

    c1, c2 = st.columns(2)
    set_start = c1.button("Set start", width='stretch')
    set_end = c2.button("Set end", width='stretch')

    viewbox = corridor_viewbox_ca(meta_ml, pad_deg=0.2)
    WARN_KM = 20.0

    if set_start and start_q.strip():
        viewbox = LA_SD_VIEWBOX
        cands = geocode_us_candidates_with_fallback(
            start_q.strip(), viewbox=LA_SD_VIEWBOX, limit=5)

        if not cands:
            st.error(
                "No match found inside California I-5 corridor. "
                "Try adding city (e.g. 'Madero, Mission Viejo')."
            )
        else:
            options = [c["display_name"] for c in cands]
            chosen = st.selectbox(
                "Matched start location",
                options,
                index=0,
                key="start_geo_pick"
            )
            c = cands[options.index(chosen)]

            sid, dist_km, _ = nearest_station_ml(meta_ml, c["lat"], c["lon"])
            st.session_state.origin_label = station_to_label_ml[sid]
            st.session_state["origin_station_selectbox"] = st.session_state.origin_label
            st.rerun()

            st.success(f"Start set to station {sid} (~{dist_km:.1f} km)")
            if dist_km > WARN_KM:
                st.warning(
                    f"Nearest station is > {WARN_KM:.0f} km away — "
                    "location may be off the corridor."
                )

    if set_end and end_q.strip():
        viewbox = LA_SD_VIEWBOX
        cands = geocode_us_candidates_with_fallback(
            end_q.strip(), viewbox=LA_SD_VIEWBOX, limit=5)

        if not cands:
            st.error(
                "No match found inside California I-5 corridor. "
                "Try adding city (e.g. 'Madero, Mission Viejo')."
            )
        else:
            options = [c["display_name"] for c in cands]
            chosen = st.selectbox(
                "Matched end location",
                options,
                index=0,
                key="end_geo_pick"
            )
            c = cands[options.index(chosen)]

            sid, dist_km, _ = nearest_station_ml(meta_ml, c["lat"], c["lon"])
            st.session_state.dest_label = station_to_label_ml[sid]
            st.session_state["dest_station_selectbox"] = st.session_state.dest_label
            st.rerun()

            st.success(f"End set to station {sid} (~{dist_km:.1f} km)")
            if dist_km > WARN_KM:
                st.warning(
                    f"Nearest station is > {WARN_KM:.0f} km away — "
                    "location may be off the corridor."
                )

# ML-only dropdowns (use session state so search updates the selection)
origin_label = st.sidebar.selectbox(
    "Origin station",
    labels_ml,
    index=labels_ml.index(st.session_state.origin_label),
    key="origin_station_selectbox",
)

dest_label = st.sidebar.selectbox(
    "Destination station",
    labels_ml,
    index=labels_ml.index(st.session_state.dest_label),
    key="dest_station_selectbox",
)

# keep state synced with manual dropdown changes
st.session_state.origin_label = origin_label
st.session_state.dest_label = dest_label

origin = int(label_to_station_ml[origin_label])
dest = int(label_to_station_ml[dest_label])


# sanity: Abs_PM should increase northbound
o_pm = float(meta_disp.loc[meta_disp["Station"] == origin, "Abs_PM"].iloc[0])
d_pm = float(meta_disp.loc[meta_disp["Station"] == dest, "Abs_PM"].iloc[0])

i, j = idx_of_station(origin), idx_of_station(dest)
if i == j:
    st.sidebar.error("Origin and destination are the same station.")
    st.stop()
if i > j:
    st.sidebar.warning(
        "Destination is south of origin along NB ordering. Swapping.")
    origin, dest = dest, origin
    i, j = j, i

day_type = st.sidebar.radio(
    "Day type", ["Weekday", "Weekend"], horizontal=True)
weekday = (day_type == "Weekday")

times = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
time_str = st.sidebar.selectbox(
    "Departure time (30-min bins)", times, index=32)  # default 16:00
min_coverage = 0.90
with st.sidebar.expander("Advanced Settings", expanded=False):

    min_coverage = st.slider(
        "Min station coverage per timestamp",
        min_value=0.50,
        max_value=1.00,
        value=0.90,
        step=0.01,
        help="Fraction of stations that must have data at a timestamp"
    )

    layout_mode = st.radio(
        "Layout",
        ["Map first (top), Results below", "Side-by-side (Results | Map)"],
        index=0,  # ✅ default map-first
        help="Choose how results and map are arranged"
    )

    downsample_step = st.slider(
        "Map detail (lower = more detail)",
        min_value=1,
        max_value=6,
        value=2,
        step=1,
        help="Downsample corridor stations for faster rendering"
    )

    jump_km = st.slider(
        "Max gap to connect (km)",
        min_value=2.0,
        max_value=20.0,
        value=8.0,
        step=0.5,
        help="Break map segments if stations are too far apart"
    )
# ---------------------- Compute ----------------------
dfT = compute_T_for_corridor(df, ordered, origin, dest, float(min_coverage))


@st.cache_data(show_spinner=False)
def station_bucket_speed(df, corridor_stations, weekday: bool, tod_label: str):
    d = df[df["Station"].isin(corridor_stations)].copy()
    d = d.dropna(subset=["Timestamp", "AvgSpeed"])
    d["weekday"] = d["Timestamp"].dt.weekday < 5
    minutes = d["Timestamp"].dt.hour * 60 + d["Timestamp"].dt.minute
    d["tod_bin"] = (minutes // 30).astype(int)
    d["tod_label"] = d["tod_bin"].apply(
        lambda b: f"{(b*30)//60:02d}:{(b*30)%60:02d}")

    b = d[(d["weekday"] == weekday) & (d["tod_label"] == tod_label)]
    g = (b.groupby("Station", as_index=False)
         .agg(mean_speed=("AvgSpeed", "mean"),
              p10_speed=("AvgSpeed", lambda s: s.quantile(0.10)),
              p50_speed=("AvgSpeed", "median"),
              p90_speed=("AvgSpeed", lambda s: s.quantile(0.90)),
              n=("AvgSpeed", "size")))
    return g


def make_corridor_map(df, meta_disp, ordered, i, j, weekday, time_str,
                      downsample_step: int = 2,
                      jump_km: float = 8.0):
    corridor_stations = ordered[i:j+1]

    # ---- per-station speed stats for chosen bucket
    speed_stats = station_bucket_speed(
        df, corridor_stations, weekday, time_str)

    # ---- meta subset with required columns
    m = meta_disp.copy()

    # ensure canonical columns exist
    for col in ["Station", "Freeway", "Direction", "LaneType", "Latitude", "Longitude", "Name", "Abs_PM"]:
        if col not in m.columns:
            # okay if Abs_PM missing in the display DF, but lat/lon must exist for map
            if col in ["Latitude", "Longitude", "Station"]:
                return None

    # hard corridor filter
    if "Freeway" in m.columns:
        m = m[pd.to_numeric(m["Freeway"], errors="coerce") == 5]
    if "Direction" in m.columns:
        m = m[m["Direction"].astype(str).str.upper().str.strip() == "N"]
    if "LaneType" in m.columns:
        m = m[m["LaneType"].astype(str).str.upper().str.strip() == "ML"]

    m = m.dropna(subset=["Station", "Latitude", "Longitude"]).copy()
    m["Station"] = pd.to_numeric(m["Station"], errors="coerce").astype("Int64")
    m = m.dropna(subset=["Station"]).copy()
    m["Station"] = m["Station"].astype(int)

    # merge speed info
    m = m.merge(speed_stats, on="Station", how="left")

    # keep only corridor slice in correct order
    order_pos = {sid: k for k, sid in enumerate(corridor_stations)}
    m["order"] = m["Station"].map(order_pos)
    m = m.dropna(subset=["order"]).sort_values("order").copy()

    if len(m) < 2:
        return None

    # optional downsample to reduce traces & make map faster
    if downsample_step > 1:
        m = m.iloc[::downsample_step].copy()
        if m["Station"].iloc[-1] != corridor_stations[-1]:
            # ensure end included
            m = pd.concat([m, m.iloc[[-1]]], ignore_index=True)

    # fill missing mean_speed with corridor median (or 60 mph)
    if m["mean_speed"].notna().any():
        fill = float(m["mean_speed"].median())
    else:
        fill = 60.0
    m["mean_speed"] = m["mean_speed"].fillna(fill)

    lats = m["Latitude"].to_numpy(float)
    lons = m["Longitude"].to_numpy(float)
    speeds = m["mean_speed"].to_numpy(float)
    names = m["Name"].astype(str).fillna("").to_numpy()

    # ---- Build segments (k -> k+1), drop huge jumps
    segs = []
    for k in range(len(m) - 1):
        lat1, lon1 = lats[k], lons[k]
        lat2, lon2 = lats[k+1], lons[k+1]
        dist = haversine_km(lat1, lon1, lat2, lon2)
        if dist > jump_km:
            # break route at large discontinuities (ramps/metadata glitches)
            continue
        v = 0.5 * (speeds[k] + speeds[k+1])  # segment speed estimate
        segs.append((k, lat1, lon1, lat2, lon2, v))

    if len(segs) < 2:
        return None

    # ---- Map speed -> color (red slow, green fast)
    # Use corridor-specific quantiles for good contrast
    vvals = np.array([s[-1] for s in segs], dtype=float)
    v_lo = float(np.quantile(vvals, 0.05))
    v_hi = float(np.quantile(vvals, 0.95))
    if v_hi <= v_lo:
        v_lo, v_hi = float(vvals.min()), float(vvals.max() + 1e-6)

    def speed_to_color(v):
        # normalize to [0,1] where 0=slow (red) and 1=fast (green)
        t = (v - v_lo) / (v_hi - v_lo + 1e-9)
        t = float(np.clip(t, 0.0, 1.0))
        # RdYlGn: 0=red, 1=green (so use t directly)
        return sample_colorscale("RdYlGn", [t])[0]

    # ---- Center/zoom fit
    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_c = (lat_min + lat_max) / 2
    lon_c = (lon_min + lon_max) / 2
    span = max(lat_max - lat_min, lon_max - lon_min)
    if span < 0.1:
        zoom = 11
    elif span < 0.3:
        zoom = 9.5
    elif span < 0.6:
        zoom = 8.2
    else:
        zoom = 7.4

    fig = go.Figure()

    # ---- Add colored segments
    for (k, lat1, lon1, lat2, lon2, v) in segs:
        color = speed_to_color(v)
        hover = (
            f"<b>Segment</b><br>"
            f"{names[k]} → {names[k+1]}<br>"
            f"Expected speed: <b>{v:.1f} mph</b><br>"
            f"Day: {'Weekday' if weekday else 'Weekend'} | Bin: {time_str}"
        )
        fig.add_trace(go.Scattermap(
            lat=[lat1, lat2],
            lon=[lon1, lon2],
            mode="lines",
            line=dict(width=6, color=color),
            hoverinfo="text",
            text=hover,
            showlegend=False,
        ))

    # ---- Start/end markers
    fig.add_trace(go.Scattermap(
        lat=[lats[0]], lon=[lons[0]],
        mode="markers+text",
        marker=dict(size=16, color="#22c55e"),
        text=["START"],
        textposition="top right",
        showlegend=False,
    ))
    fig.add_trace(go.Scattermap(
        lat=[lats[-1]], lon=[lons[-1]],
        mode="markers+text",
        marker=dict(size=16, color="#ef4444"),
        text=["END"],
        textposition="top right",
        showlegend=False,
    ))

    # ---- Optional: stations (faint)
    # You can keep your radio toggle to add these as a separate trace:
    fig.add_trace(go.Scattermap(
        lat=lats,
        lon=lons,
        mode="markers",
        marker=dict(size=6, opacity=0.35),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=lat_c, lon=lon_c),
            zoom=zoom,
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=520,
    )

    return fig


# def make_corridor_map(df, meta_disp, ordered, i, j, weekday, time_str):
#     corridor_stations = ordered[i:j+1]

#     speed_stats = station_bucket_speed(
#         df, corridor_stations, weekday, time_str)

#     need = ["Station", "Name", "District", "Freeway",
#             "Direction", "LaneType", "Latitude", "Longitude"]
#     have = [c for c in need if c in meta_disp.columns]
#     meta_map = meta_disp[have].copy()

#     if "Latitude" not in meta_map.columns or "Longitude" not in meta_map.columns:
#         return None
#     meta_map = meta_map.dropna(subset=["Latitude", "Longitude"])
#     meta_map["Station"] = meta_map["Station"].astype(int)

#     m = meta_map.merge(speed_stats, on="Station", how="left")

#     order_pos = {sid: k for k, sid in enumerate(corridor_stations)}
#     m["order"] = m["Station"].map(order_pos)
#     m = m.dropna(subset=["order"]).sort_values("order")

#     if len(m) < 2:
#         return None

#     # Fill missing mean_speed to avoid crashes
#     m["mean_speed"] = m["mean_speed"].fillna(
#         m["mean_speed"].median() if m["mean_speed"].notna().any() else 60.0
#     )
#     m["n"] = m["n"].fillna(0).astype(int)

#     FREEFLOW = 70.0
#     m["slowness"] = np.clip(1.0 - (m["mean_speed"] / FREEFLOW), 0, 1)

#     COLORSCALE = "RdYlGn_r"

#     def cs_color(v01: float) -> str:
#         v01 = float(np.clip(v01, 0.0, 1.0))
#         return sample_colorscale(COLORSCALE, [v01])[0]

#     lats = m["Latitude"].to_numpy()
#     lons = m["Longitude"].to_numpy()
#     speeds = m["mean_speed"].to_numpy()
#     slow = m["slowness"].to_numpy()
#     names = m["Name"].fillna("(no name)").to_numpy()
#     stations = m["Station"].to_numpy()
#     ns = m["n"].to_numpy()

#     lat_c = float(np.mean(lats))
#     lon_c = float(np.mean(lons))

#     fig = go.Figure()

#     # per-segment colored line
#     for k in range(len(m) - 1):
#         v = 0.5 * (slow[k] + slow[k+1])
#         fig.add_trace(go.Scattermap(
#             lat=[lats[k], lats[k+1]],
#             lon=[lons[k], lons[k+1]],
#             mode="lines",
#             line=dict(width=6, color=cs_color(v)),
#             hoverinfo="text",
#             text=(
#                 f"Segment: {names[k]} → {names[k+1]}<br>"
#                 f"Mean speed: {0.5*(speeds[k]+speeds[k+1]):.1f} mph<br>"
#                 f"Bucket n: {ns[k]} / {ns[k+1]}"
#             ),
#             showlegend=False
#         ))

#     # station markers
#     fig.add_trace(go.Scattermap(
#         lat=lats,
#         lon=lons,
#         mode="markers",
#         marker=dict(
#             size=7,
#             color=speeds,
#             colorscale="Viridis",
#             showscale=True,
#             colorbar=dict(title="Mean speed (mph)"),
#         ),
#         hoverinfo="text",
#         text=[
#             f"{names[k]}<br>ID={stations[k]}<br>Mean speed={speeds[k]:.1f} mph<br>Bucket n={ns[k]}"
#             for k in range(len(m))
#         ],
#         showlegend=False
#     ))

#     # Start/End markers
#     fig.add_trace(go.Scattermap(
#         lat=[lats[0]], lon=[lons[0]],
#         mode="markers+text",
#         marker=dict(size=12, color="#22c55e"),
#         text=["START"], textposition="top right",
#         showlegend=False
#     ))
#     fig.add_trace(go.Scattermap(
#         lat=[lats[-1]], lon=[lons[-1]],
#         mode="markers+text",
#         marker=dict(size=12, color="#ef4444"),
#         text=["END"], textposition="top right",
#         showlegend=False
#     ))


#     fig.update_layout(
#         mapbox=dict(
#             style="open-street-map",
#             center=dict(lat=lat_c, lon=lon_c),
#             zoom=7.5,
#         ),
#         margin=dict(l=10, r=10, t=10, b=10),
#         height=520,
#     )
#     return fig
# --- Time-of-day curve stats (for this corridor) ---
tod_curve = (
    dfT.dropna(subset=["T"])
       .groupby(["weekday", "tod_label"], sort=True)
       .agg(
        mean_min=("T", "mean"),
        p95_min=("T", lambda s: s.quantile(0.95)),
        n=("T", "size"),
    )
    .reset_index()
)

tod_curve["day_type"] = np.where(tod_curve["weekday"], "Weekday", "Weekend")

# Ensure correct chronological order for labels like "00:00", "00:30", ...
tod_curve["tod_bin"] = tod_curve["tod_label"].apply(
    lambda t: int(t[:2]) * 2 + (1 if t[3:] == "30" else 0))
tod_curve = tod_curve.sort_values(["day_type", "tod_bin"])

bucket = dfT[(dfT["weekday"] == weekday) & (dfT["tod_label"] == time_str)]
x = bucket["T"].dropna()

if len(x) < 10:
    st.warning(
        f"Too few samples in this bucket (n={len(x)}). Try another time/day or lower coverage in Advanced.")
    st.stop()

# ---------------------- Core stats ----------------------
n = len(x)
mu = float(x.mean())
sd = float(x.std(ddof=1))
q50 = float(x.quantile(0.50))
q90 = float(x.quantile(0.90))
q95 = float(x.quantile(0.95))
q99 = float(x.quantile(0.99))


map_fig = make_corridor_map(df, meta_disp, ordered, i, j, weekday, time_str,
                            downsample_step=downsample_step,
                            jump_km=jump_km)

z = norm.ppf(0.975)
ci_lo = mu - z * sd / np.sqrt(n)
ci_hi = mu + z * sd / np.sqrt(n)

bti = (q95 - mu) / mu if mu > 0 else np.nan

# ---------------------- Main stats layout ----------------------
# st.markdown(
# f"**{day_type} @ {time_str}** · Corridor stations: **{j - i + 1}** · Bucket n: **{len(x)}**")


if layout_mode.startswith("Side-by-side"):
    left, right = st.columns([1.05, 1.35], gap="large")

    with left:
        st.subheader("Results")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean (min)", f"{mu:.1f}")
        c2.metric("Std (min)", f"{sd:.1f}")
        c3.metric("Median (min)", f"{q50:.1f}")
        c4.metric("Samples (n)", f"{n}")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("p90 (min)", f"{q90:.1f}")
        d2.metric("p95 (min)", f"{q95:.1f}")
        d3.metric("p99 (min)", f"{q99:.1f}")
        d4.metric("BTI", f"{bti:.3f}")

        e1, e2 = st.columns(2)
        with e1:
            st.write("**95% CI for mean (CLT)**")
            st.write(f"[{ci_lo:.1f}, {ci_hi:.1f}] minutes")
        with e2:
            st.write("**Quick interpretation**")
            st.write(
                f"Plan around **{q95:.0f} min** to be ~95% safe in this bucket.")

    with right:
        st.subheader("Map")
        if map_fig is None:
            st.warning(
                "Not enough stations with lat/lon metadata to draw the corridor.")
        else:
            st.plotly_chart(map_fig, width='stretch')
            # st.caption(
            # "Route color indicates expected slowness in the selected bucket (red = slower).")
else:
    st.subheader("Map")
    if map_fig is None:
        st.warning(
            "Not enough stations with lat/lon metadata to draw the corridor.")
    else:
        st.plotly_chart(map_fig, width='stretch')
        # st.caption(
        # "Route color indicates expected slowness in the selected bucket (red = slower).")

    # st.divider()

    st.subheader("Results", width="content")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean (min)", f"{mu:.1f}")
    c2.metric("Std (min)", f"{sd:.1f}")
    c3.metric("Median (min)", f"{q50:.1f}")
    c4.metric("Samples (n)", f"{n}")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("p90 (min)", f"{q90:.1f}")
    d2.metric("p95 (min)", f"{q95:.1f}")
    d3.metric("p99 (min)", f"{q99:.1f}")
    d4.metric("BTI", f"{bti:.3f}")

    e1, e2 = st.columns(2)
    with e1:
        st.write("**95% CI for mean (CLT)**")
        st.write(f"[{ci_lo:.1f}, {ci_hi:.1f}] minutes")
    with e2:
        st.write("**Quick interpretation**")
        st.write(
            f"Plan around **{q95:.0f} min** to be ~95% safe in this bucket.")


# ---------------------- Collapsed lateness block ----------------------
tau = None
with st.expander("Lateness / reliability (optional)", expanded=False):
    late_mode = st.radio(
        "Lateness definition",
        ["Buffer above mean", "Absolute threshold"],
        horizontal=True,
        help="Buffer mode is intuitive across different corridor lengths."
    )

    if late_mode == "Buffer above mean":
        buffer_min = st.slider("Buffer above mean (minutes)", 0, 60, 20)
        tau = mu + buffer_min
        st.caption(f"Using τ = mean + buffer = {tau:.1f} minutes.")
    else:
        lo = int(max(0, np.floor(x.quantile(0.01) - 10)))
        hi = int(np.ceil(x.quantile(0.99) + 20))
        default = int(np.ceil(x.quantile(0.95)))
        tau = st.slider("Absolute threshold (minutes)", lo, hi, default)

    p_late = float((x > tau).mean())
    cdf_at_tau = float((x <= tau).mean())
    st.write(f"**P(T > τ)** ≈ {p_late:.3f}")
    st.write(f"Empirical CDF at τ: **F(τ) ≈ {cdf_at_tau:.3f}**")

# ---------------------- Prettier plots in tabs ----------------------
st.subheader("Plots")
tab1, tab2, tab3 = st.tabs(["Time-of-day curve", "CDF", "Histogram / PDF"])

# Shared x-range
xmin = max(0.0, float(x.quantile(0.01) - 10))
xmax = float(x.quantile(0.99) + 10)


def add_vline(fig, x0, name, dash="solid"):
    fig.add_vline(
        x=x0,
        line_width=2,
        line_dash=dash,
        annotation_text=name,
        annotation_position="top",
        opacity=0.9,
    )


with tab1:
    # Plot mean and p95 across time-of-day for BOTH weekday & weekend
    fig = go.Figure()

    # Use numeric x = tod_bin (0..47). We'll label ticks as HH:MM.
    tickvals = list(range(0, 48, 2))  # every hour
    ticktext = [f"{h:02d}:00" for h in range(24)]

    for day_name in ["Weekday", "Weekend"]:
        dd = tod_curve[tod_curve["day_type"] == day_name]
        if dd.empty:
            continue

        fig.add_trace(go.Scatter(
            x=dd["tod_bin"],
            y=dd["mean_min"],
            mode="lines+markers",
            name=f"{day_name} mean",
            hovertemplate="Bin=%{x} (%{customdata})<br>Mean=%{y:.1f} min<br>n=%{text}<extra></extra>",
            customdata=dd["tod_label"],
            text=dd["n"],
        ))

        fig.add_trace(go.Scatter(
            x=dd["tod_bin"],
            y=dd["p95_min"],
            mode="lines",
            line=dict(dash="dash"),
            name=f"{day_name} p95",
            hovertemplate="Bin=%{x} (%{customdata})<br>p95=%{y:.1f} min<br>n=%{text}<extra></extra>",
            customdata=dd["tod_label"],
            text=dd["n"],
        ))

    # Highlight currently selected day/time using numeric bin
    selected_bin = int(time_str[:2]) * 2 + (1 if time_str[3:] == "30" else 0)

    # A robust vertical line as a shape on numeric axes
    fig.add_shape(
        type="line",
        x0=selected_bin, x1=selected_bin,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(width=2, dash="dot"),
    )
    fig.add_annotation(
        x=selected_bin, y=1.02,
        xref="x", yref="paper",
        text=f"selected {('Weekday' if weekday else 'Weekend')} {time_str}",
        showarrow=False
    )

    fig.update_layout(
        template="plotly_white",
        title="Conditional travel time vs departure time (mean and p95)",
        xaxis_title="Departure time (30-min bins)",
        yaxis_title="Travel time (minutes)",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="left", x=0),
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.06)",
        range=[-0.5, 47.5],
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")

    st.plotly_chart(fig, width='stretch')

    st.caption(
        "This curve estimates conditional expectation E[T | time-of-day, weekday/weekend] "
        "and the conditional 95th percentile (tail risk)."
    )

with tab2:
    # Empirical CDF
    xs = np.sort(x.values)
    ys = np.arange(1, len(xs) + 1) / len(xs)

    cdf = go.Figure()
    cdf.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(width=3),
        name="Empirical CDF"
    ))

    cdf.update_layout(
        template="plotly_white",
        title="Empirical CDF",
        xaxis_title="Travel time (minutes)",
        yaxis_title="F_T(t)",
        margin=dict(l=20, r=20, t=50, b=20),
        height=380,
        yaxis=dict(range=[0, 1]),
    )
    cdf.update_xaxes(range=[xmin, xmax], showgrid=True,
                     gridcolor="rgba(0,0,0,0.08)")
    cdf.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")

    add_vline(cdf, mu, "mean", dash="solid")
    add_vline(cdf, q95, "p95", dash="dash")
    if tau is not None:
        add_vline(cdf, tau, "τ", dash="dot")

    st.plotly_chart(cdf, width='stretch')

with tab3:
    hist = px.histogram(
        x,
        nbins=35,
        histnorm="probability density",
        opacity=0.85,
        labels={"value": "Travel time (minutes)"},
        title="Empirical travel-time distribution",
    )
    hist.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=420,
    )
    hist.update_xaxes(range=[xmin, xmax], showgrid=True,
                      gridcolor="rgba(0,0,0,0.08)")
    hist.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")

    add_vline(hist, mu, f"mean={mu:.1f}", dash="solid")
    add_vline(hist, q95, f"p95={q95:.1f}", dash="dash")
    if tau is not None:
        add_vline(hist, tau, f"τ={tau:.1f}", dash="dot")

    # --- Rug plot (raw samples along x-axis) ---
    # downsample a bit if huge (keeps UI snappy)
    rug = x.values
    if len(rug) > 2000:
        rug = np.random.default_rng(0).choice(rug, size=2000, replace=False)

    # place rug slightly below the x-axis baseline
    # we’ll extend y-range slightly so the rug is visible
    y_max = float(hist.data[0]["y"].max()) if len(
        hist.data) > 0 and hist.data[0]["y"] is not None else None
    if y_max is not None:
        hist.update_yaxes(range=[-0.06 * y_max, 1.05 * y_max])

    hist.add_trace(go.Scatter(
        x=rug,
        y=np.full_like(
            rug, -0.03 * (y_max if y_max is not None else 1.0), dtype=float),
        mode="markers",
        marker=dict(size=6, opacity=0.25),
        name="samples (rug)",
        hoverinfo="skip",
        showlegend=False,
    ))

    st.plotly_chart(hist, width='stretch')

# ----------------------  selection / coverage ( collapsed) ----------------------
# Collapsed details
with st.expander("Debug Info (selection + data coverage)", expanded=False):
    cL, cR = st.columns([2, 1])
    with cL:
        st.write(f"**Origin:** {origin_label}")
        st.write(f"**Destination:** {dest_label}")
        st.write(f"**Corridor stations included:** {j - i + 1}")
        st.write(f"**Condition:** {day_type}, departure bin **{time_str}**")
        st.write(
            f"AbsPM origin={o_pm:.2f} → dest={d_pm:.2f} (NB should increase)")
    with cR:
        st.write(f"Bucket samples: **{len(x)}**")
        st.write(f"Total corridor samples: **{len(dfT)}**")
        st.write(f"Coverage threshold: **{min_coverage:.2f}**")
