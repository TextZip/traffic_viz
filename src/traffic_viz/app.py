import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
# import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import requests
# from functools import lru_cache
from sklearn.mixture import GaussianMixture
try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except Exception:
    HAS_HMM = False

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

st.title("I-5 Northbound Travel Time Predictor", text_alignment="center")
st.caption(
    "Pick origin/destination stations by name to get probabilistic travel-time stats.", text_alignment="center")
# Create left/right spacers to center the button group
st.markdown(
    """
    <div style="text-align: center; margin-top: 0.5rem;">
        🔗 <a href="https://github.com/TextZip/traffic_viz" target="_blank">Code</a>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        📄 <a href="https://your-report-link.com" target="_blank">Report</a>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        📂 <a href="https://drive.google.com/drive/folders/1Ms151kBdxH-d284sY8oyMqVU61OwpqXG?usp=sharing" target="_blank">Dataset</a>
    </div>
    """,
    unsafe_allow_html=True,
)

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


@st.cache_data(show_spinner=False)
def fit_gmm_cached(
    x_values: np.ndarray,
    k_mode: str,
    K: int,
    k_max: int,
    seed: int,
):
    x_np = np.asarray(x_values, dtype=float)
    if x_np.size < 30:
        return None, None, None

    X = x_np.reshape(-1, 1)

    if k_mode.startswith("Auto"):
        best = None
        best_bic = np.inf
        bestK = None

        for Kcand in range(1, int(k_max) + 1):
            model = fit_gmm_1d(x_np, Kcand, seed=seed)
            bic = model.bic(X)
            if bic < best_bic:
                best_bic = bic
                best = model
                bestK = Kcand

        return best, bestK, float(best_bic)

    # Manual
    model = fit_gmm_1d(x_np, int(K), seed=seed)
    bic = model.bic(X)
    return model, int(K), float(bic)


@st.cache_data(show_spinner=False)
def fit_hmm_daily_sequences(
    dfT_in: pd.DataFrame,
    weekday_flag: bool,
    n_states: int,
    seed: int = 0,
    min_bins_per_day: int = 40,   # require at least this many of 48 bins
    use_residuals: bool = True,   # recommended
):
    """
    Option B HMM: train on per-day sequences of length 48 (30-min bins).
    Fits separate model for weekday_flag in caller.

    Returns:
      hmm: fitted GaussianHMM
      baseline: np.ndarray shape (48,) mean curve (if use_residuals else zeros)
      day_mat: pd.DataFrame shape (num_days, 48) filled travel times
      gamma_mat: np.ndarray shape (num_days, 48, n_states) posterior probs
    """
    if not HAS_HMM:
        return None, None, None, None

    d = dfT_in.dropna(subset=["T", "tod_bin", "weekday"]).copy()
    d = d[d["weekday"] == weekday_flag].copy()
    if len(d) < 48 * 10:  # need at least ~10 days worth of data
        return None, None, None, None

    d = d.sort_index()
    d["date"] = d.index.date

    # day x tod_bin matrix
    mat = (
        d.pivot_table(index="date", columns="tod_bin",
                      values="T", aggfunc="mean")
        .reindex(columns=range(48))
        .sort_index()
    )

    # keep days with enough bins
    valid = mat.notna().sum(axis=1) >= int(min_bins_per_day)
    mat = mat.loc[valid].copy()
    if mat.shape[0] < 10:
        return None, None, None, None

    # fill remaining missing bins (interpolate across time-of-day)
    mat = mat.interpolate(axis=1, limit_direction="both")

    if use_residuals:
        baseline = mat.mean(axis=0).to_numpy(dtype=float)  # (48,)
        resid_mat = mat.to_numpy(dtype=float) - baseline[None, :]
    else:
        baseline = np.zeros(48, dtype=float)
        resid_mat = mat.to_numpy(dtype=float)

    # stack sequences for hmmlearn
    X = resid_mat.reshape(-1, 1)
    lengths = [48] * resid_mat.shape[0]

    hmm = GaussianHMM(
        n_components=int(n_states),
        covariance_type="diag",
        n_iter=300,
        tol=1e-3,
        random_state=int(seed),
        init_params="stmc",
    )
    hmm.fit(X, lengths=lengths)

    gamma = hmm.predict_proba(X)  # (num_days*48, n_states)
    gamma_mat = gamma.reshape(resid_mat.shape[0], 48, int(n_states))

    return hmm, baseline, mat, gamma_mat


# ------------------ HMM HELPERS --------------------------------
def mix_cdf(t: np.ndarray, w: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    t = np.asarray(t, float).reshape(-1)
    w = np.asarray(w, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    z = (t[:, None] - mu[None, :]) / (sigma[None, :] + 1e-12)
    return np.sum(w[None, :] * norm.cdf(z), axis=1)


def mix_pdf(t: np.ndarray, w: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    t = np.asarray(t, float).reshape(-1)
    w = np.asarray(w, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    z = (t[:, None] - mu[None, :]) / (sigma[None, :] + 1e-12)
    return np.sum(w[None, :] * (norm.pdf(z) / (sigma[None, :] + 1e-12)), axis=1)


def mix_mean_var(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    w = np.asarray(w, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    m = float(np.sum(w * mu))
    second = float(np.sum(w * (sigma**2 + mu**2)))
    v = max(0.0, second - m*m)
    return m, v


def mix_quantile(q: float, w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, lo: float, hi: float, iters: int = 80):
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0

    # expand bracket if needed
    c_lo = mix_cdf(np.array([lo]), w, mu, sigma)[0]
    c_hi = mix_cdf(np.array([hi]), w, mu, sigma)[0]
    step = max(5.0, 0.25 * (hi - lo))
    for _ in range(40):
        if c_lo <= q <= c_hi:
            break
        if c_hi < q:
            hi += step
            c_hi = mix_cdf(np.array([hi]), w, mu, sigma)[0]
        if c_lo > q:
            lo = max(0.0, lo - step)
            c_lo = mix_cdf(np.array([lo]), w, mu, sigma)[0]
        step *= 1.5

    # bisection
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        c = mix_cdf(np.array([mid]), w, mu, sigma)[0]
        if c < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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

default_origin = int(meta_disp[meta_disp["District"] == 11]["Station"].iloc[0])
default_dest = int(meta_disp[meta_disp["District"] == 7]["Station"].iloc[-1])
origin_label_default = meta_disp[meta_disp["Station"]
                                 == default_origin]["label"].iloc[0]
dest_label_default = meta_disp[meta_disp["Station"]
                               == default_dest]["label"].iloc[0]


def idx_of_station(st_id: int) -> int:
    return ordered.index(st_id)

# -----------------------GMM Helpers ----------------------------


def fit_gmm_1d(x: np.ndarray, K: int, seed: int = 0):
    """
    Fit a 1D Gaussian Mixture on x (shape [n]).
    Returns fitted sklearn GaussianMixture.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=int(K),
        covariance_type="full",
        # slightly stronger regularization (traffic has outliers)
        reg_covar=1e-3,
        n_init=10,               # more inits -> fewer bad local optima
        init_params="kmeans",    # explicit (default), good for 1D
        max_iter=300,            # avoid early stopping on hard buckets
        tol=1e-3,                # convergence threshold
        random_state=int(seed),
    )
    gmm.fit(x)
    return gmm


def gmm_mean_var_1d(gmm: GaussianMixture):
    w = gmm.weights_.astype(float)
    mu = gmm.means_.ravel().astype(float)
    var_k = gmm.covariances_.reshape(-1).astype(float)

    mean = float(np.sum(w * mu))
    second = float(np.sum(w * (var_k + mu**2)))
    var = max(0.0, second - mean**2)  # numerical safety
    return mean, var


def gmm_cdf_1d(gmm: GaussianMixture, t: np.ndarray):
    t = np.asarray(t, dtype=float).reshape(-1)
    w = gmm.weights_.astype(float)
    mu = gmm.means_.ravel().astype(float)
    var_k = gmm.covariances_.reshape(-1).astype(float)
    sigma = np.sqrt(var_k + 1e-12)

    z = (t[:, None] - mu[None, :]) / sigma[None, :]
    return np.sum(w[None, :] * norm.cdf(z), axis=1)


def gmm_pdf_1d(gmm: GaussianMixture, t: np.ndarray):
    t = np.asarray(t, dtype=float).reshape(-1)
    w = gmm.weights_.astype(float)
    mu = gmm.means_.ravel().astype(float)
    var_k = gmm.covariances_.reshape(-1).astype(float)
    sigma = np.sqrt(var_k + 1e-12)

    z = (t[:, None] - mu[None, :]) / sigma[None, :]
    return np.sum(w[None, :] * (norm.pdf(z) / sigma[None, :]), axis=1)


def gmm_quantile_1d(gmm: GaussianMixture, q: float, lo: float, hi: float, iters: int = 80):
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0

    # Expand bracket until it contains q
    c_lo = gmm_cdf_1d(gmm, np.array([lo]))[0]
    c_hi = gmm_cdf_1d(gmm, np.array([hi]))[0]

    # If bracket doesn't contain q, expand outward
    # (cap expansions to avoid infinite loops)
    step = max(5.0, 0.25 * (hi - lo))
    for _ in range(40):
        if c_lo <= q <= c_hi:
            break
        if c_hi < q:
            hi += step
            c_hi = gmm_cdf_1d(gmm, np.array([hi]))[0]
        if c_lo > q:
            lo = max(0.0, lo - step)
            c_lo = gmm_cdf_1d(gmm, np.array([lo]))[0]
        step *= 1.5

    # Bisection
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        c = gmm_cdf_1d(gmm, np.array([mid]))[0]
        if c < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gmm_components_sorted(gmm: GaussianMixture):
    w = gmm.weights_.astype(float)
    mu = gmm.means_.ravel().astype(float)
    var_k = gmm.covariances_.reshape(-1).astype(float)
    idx = np.argsort(mu)
    return w[idx], mu[idx], np.sqrt(var_k[idx] + 1e-12)


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

    # viewbox = corridor_viewbox_ca(meta_ml, pad_deg=0.2)
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
with st.sidebar.expander("Advanced", expanded=False):

    st.subheader("Data Processing Settings")
    min_coverage = st.slider(
        "Min station coverage per timestamp",
        min_value=0.50,
        max_value=1.00,
        value=0.90,
        step=0.01,
        help="Fraction of stations that must have data at a timestamp"
    )

    st.subheader("UI Settings")
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

    st.subheader("GMM Settings")
    use_gmm = st.checkbox("Enable GMM", value=False)
    k_mode = st.radio("Choose K", ["Auto (BIC)", "Manual"], horizontal=True)

    # Always define both, so downstream code never crashes
    K = 2
    k_max = 6

    if k_mode == "Manual":
        K = st.slider("Components (K)", 1, 6, 2)
    else:
        k_max = st.slider("Max K (BIC search)", 1, 8, 6)

    st.subheader("HMM Settings")
    use_hmm = st.checkbox("Enable HMM",
                          value=False, disabled=not HAS_HMM)
    n_states = st.slider("HMM states", 2, 6, 3)
    min_bins_per_day = st.slider(
        "Min bins/day (quality filter)", 24, 48, 40, step=1)
    use_residuals = st.checkbox(
        "Residualize by time-of-day baseline", value=True)
    if use_hmm and not HAS_HMM:
        st.sidebar.warning(
            "hmmlearn not installed. Install: pip install hmmlearn")

# ---------------------- Compute ----------------------
dfT = compute_T_for_corridor(df, ordered, origin, dest, float(min_coverage))


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

# ---------------------- HMM (Option B) ----------------------
hmm = None
hmm_baseline = None
hmm_day_mat = None
hmm_gamma_mat = None

hmm_w = hmm_mu_k = hmm_sigma_k = None
hmm_mean = hmm_sd = None
hmm_q95 = hmm_q99 = None

if use_hmm:
    hmm, hmm_baseline, hmm_day_mat, hmm_gamma_mat = fit_hmm_daily_sequences(
        dfT_in=dfT,
        weekday_flag=weekday,         # separate model for weekday/weekend
        n_states=n_states,
        seed=0,
        min_bins_per_day=min_bins_per_day,
        use_residuals=use_residuals,
    )

    if hmm is None:
        st.sidebar.warning(
            "HMM unavailable or not enough high-quality daily sequences.")
    else:
        selected_bin = int(time_str[:2]) * 2 + \
            (1 if time_str[3:] == "30" else 0)

        # state weights for this bin: average posterior across days
        w = hmm_gamma_mat[:, selected_bin, :].mean(axis=0)
        w = np.clip(w, 1e-9, None)
        w = w / w.sum()

        mu_res = hmm.means_.ravel().astype(float)
        sigma_res = np.sqrt(hmm.covars_.ravel().astype(float) + 1e-12)

        base = float(hmm_baseline[selected_bin]) if use_residuals else 0.0

        hmm_w = w
        hmm_mu_k = base + mu_res
        hmm_sigma_k = sigma_res

        hmm_mean, hmm_var = mix_mean_var(hmm_w, hmm_mu_k, hmm_sigma_k)
        hmm_sd = float(np.sqrt(hmm_var))

        x_np = x.to_numpy(dtype=float)
        lo = max(0.0, float(np.quantile(x_np, 0.001) - 15))
        hi = float(np.quantile(x_np, 0.999) + 15)
        if hi <= lo + 1e-6:
            hi = lo + 1.0

        hmm_q95 = mix_quantile(0.95, hmm_w, hmm_mu_k,
                               hmm_sigma_k, lo=lo, hi=hi)
        hmm_q99 = mix_quantile(0.99, hmm_w, hmm_mu_k,
                               hmm_sigma_k, lo=lo, hi=hi)


gmm = None
gmm_K = None
gmm_bic = None
gmm_mu = gmm_sd = None
gmm_q95 = gmm_q99 = None

if use_gmm:
    x_np = x.to_numpy(dtype=float)

    if len(x_np) < 30:
        st.sidebar.warning("Not enough samples for a stable GMM (need ~30+).")
    else:
        # ---- fit (cached) ----
        gmm, gmm_K, gmm_bic = fit_gmm_cached(
            x_values=x_np,
            k_mode=k_mode,
            K=K,
            k_max=k_max,
            seed=0,
        )

        # fit_gmm_cached can still return None if something went wrong
        if gmm is None:
            st.sidebar.warning("GMM fit failed for this bucket/settings.")
        else:
            # ---- moments ----
            gmm_mu, gmm_var = gmm_mean_var_1d(gmm)
            gmm_sd = float(np.sqrt(gmm_var))

            # ---- bracket for quantiles: empirical padded range ----
            lo = max(0.0, float(np.quantile(x_np, 0.001) - 15))
            hi = float(np.quantile(x_np, 0.999) + 15)
            if hi <= lo + 1e-6:  # extremely degenerate case
                hi = lo + 1.0

            gmm_q95 = gmm_quantile_1d(gmm, 0.95, lo=lo, hi=hi)
            gmm_q99 = gmm_quantile_1d(gmm, 0.99, lo=lo, hi=hi)


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

        if gmm is not None:
            st.markdown(f"**GMM fit:** K={gmm_K} (BIC={gmm_bic:.1f})")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("GMM mean (min)", f"{gmm_mu:.1f}")
            g2.metric("GMM std (min)", f"{gmm_sd:.1f}")
            g3.metric("GMM p95 (min)", f"{gmm_q95:.1f}")
            g4.metric("GMM p99 (min)", f"{gmm_q99:.1f}")

        if hmm_w is not None:
            st.markdown(
                f"**HMM regime model:** states={n_states} (trained on {'Weekday' if weekday else 'Weekend'} days)")
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("HMM mean (min)", f"{hmm_mean:.1f}")
            h2.metric("HMM std (min)", f"{hmm_sd:.1f}")
            h3.metric("HMM p95 (min)", f"{hmm_q95:.1f}")
            h4.metric("HMM p99 (min)", f"{hmm_q99:.1f}")

            st.caption("State weights @ this time bin: " +
                       ", ".join([f"s{k}={hmm_w[k]:.2f}" for k in range(len(hmm_w))]))

        e1, e2 = st.columns(2)
        with e1:
            st.write("**95% CI for mean (CLT)**")
            st.write(f"[{ci_lo:.1f}, {ci_hi:.1f}] minutes")
        with e2:
            st.write("**Quick interpretation**")
            plan95 = gmm_q95 if gmm is not None else q95
            st.write(
                f"Plan around **{plan95:.0f} min** to be ~95% safe in this bucket.")
            # st.write(
            # f"Plan around **{q95:.0f} min** to be ~95% safe in this bucket.")

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

    if gmm is not None:
        st.markdown(f"**GMM fit:** K={gmm_K}")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("GMM mean (min)", f"{gmm_mu:.1f}")
        g2.metric("GMM std (min)", f"{gmm_sd:.1f}")
        g3.metric("GMM p95 (min)", f"{gmm_q95:.1f}")
        g4.metric("GMM p99 (min)", f"{gmm_q99:.1f}")

    if hmm_w is not None:
        st.markdown(
            f"**HMM regime model:** states={n_states} (trained on {'Weekday' if weekday else 'Weekend'} days)")
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("HMM mean (min)", f"{hmm_mean:.1f}")
        h2.metric("HMM std (min)", f"{hmm_sd:.1f}")
        h3.metric("HMM p95 (min)", f"{hmm_q95:.1f}")
        h4.metric("HMM p99 (min)", f"{hmm_q99:.1f}")

        st.caption("State weights @ this time bin: " +
                   ", ".join([f"s{k}={hmm_w[k]:.2f}" for k in range(len(hmm_w))]))

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
    if tau is not None and gmm is not None:
        p_late_gmm = 1.0 - gmm_cdf_1d(gmm, np.array([tau]))[0]
        st.write(f"**GMM P(T > τ)** ≈ {p_late_gmm:.3f}")
    if tau is not None and hmm_w is not None:
        p_late_hmm = 1.0 - \
            mix_cdf(np.array([tau]), hmm_w, hmm_mu_k, hmm_sigma_k)[0]
        st.write(f"**HMM P(T > τ)** ≈ {p_late_hmm:.3f}")


# ---------------------- Prettier plots in tabs ----------------------
st.subheader("Plots")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Time-of-day curve", "CDF", "Histogram / PDF", "HMM Regimes"])


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

    if gmm is not None:
        grid = np.linspace(float(xmin), float(xmax), 400)
        cdf_g = gmm_cdf_1d(gmm, grid)
        cdf.add_trace(go.Scatter(
            x=grid, y=cdf_g,
            mode="lines",
            line=dict(dash="dash"),
            name="GMM CDF"
        ))

    if hmm_w is not None:
        grid = np.linspace(float(xmin), float(xmax), 400)
        cdf_h = mix_cdf(grid, hmm_w, hmm_mu_k, hmm_sigma_k)
        cdf.add_trace(go.Scatter(
            x=grid, y=cdf_h,
            mode="lines",
            line=dict(dash="dot"),
            name="HMM CDF"
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
    if gmm is not None:
        grid = np.linspace(float(xmin), float(xmax), 400)
        pdf_g = gmm_pdf_1d(gmm, grid)
        hist.add_trace(go.Scatter(
            x=grid, y=pdf_g,
            mode="lines",
            name="GMM PDF"
        ))

    if hmm_w is not None:
        grid = np.linspace(float(xmin), float(xmax), 400)
        pdf_h = mix_pdf(grid, hmm_w, hmm_mu_k, hmm_sigma_k)
        hist.add_trace(go.Scatter(
            x=grid, y=pdf_h,
            mode="lines",
            line=dict(dash="dot"),
            name="HMM PDF"
        ))
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

with tab4:
    # st.subheader("HMM Regimes (state probabilities + baseline travel time)")

    if not use_hmm:
        st.info("Enable the HMM in the sidebar to see regime plots.")
    elif hmm is None or hmm_gamma_mat is None or hmm_baseline is None:
        st.warning(
            "HMM model not available for this selection (insufficient data or missing hmmlearn).")
    else:
        # ------------------------------------------------------------
        # Data prep
        # ------------------------------------------------------------
        K_states = hmm_gamma_mat.shape[2]  # n_states
        gamma_by_bin = hmm_gamma_mat.mean(axis=0)  # (48, K)

        tod_labels = [f"{(b*30)//60:02d}:{(b*30)%60:02d}" for b in range(48)]
        tod_bin = np.arange(48)

        selected_bin = int(time_str[:2]) * 2 + \
            (1 if time_str[3:] == "30" else 0)

        # Residual emission params (means/stds)
        mu_res = hmm.means_.ravel().astype(float)  # (K,)
        sig_res = np.sqrt(hmm.covars_.ravel().astype(float) + 1e-12)  # (K,)

        # Baseline curve (travel time mean by bin) from training set
        base = np.asarray(hmm_baseline, dtype=float).reshape(-1)  # (48,)

        # State mean curves in original T space: baseline + residual_mean_k
        state_mean_curves = base[:, None] + mu_res[None, :]  # (48, K)

        # ------------------------------------------------------------
        # Plot A: Stacked area of regime probabilities over day
        # ------------------------------------------------------------
        st.markdown("### Regime occupancy over time-of-day (stacked)")

        fig_area = go.Figure()

        order = np.argsort(mu_res)  # low residual to high residual
        for k in order:
            fig_area.add_trace(go.Scatter(
                x=tod_bin,
                y=gamma_by_bin[:, k],
                name=f"state {k}",
                stackgroup="one",
                mode="none",                 # ✅ no lines
                # hoveron="fills",             # ✅ hover works on filled areas
                hovertemplate="bin=%{x} (%{customdata})<br>P(state)= %{y:.2f}<extra></extra>",
                customdata=tod_labels,
            ))
        # Highlight selected time bin
        fig_area.add_shape(
            type="line",
            x0=selected_bin, x1=selected_bin,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(width=2, dash="dot"),
        )
        fig_area.add_annotation(
            x=selected_bin, y=1.08,
            xref="x", yref="paper",
            text=f"selected {('Weekday' if weekday else 'Weekend')} {time_str}",
            showarrow=False
        )

        tickvals = list(range(0, 48, 2))
        ticktext = [f"{h:02d}:00" for h in range(24)]

        fig_area.update_layout(
            template="plotly_white",
            title=f"P(state | time-of-day) averaged over days ({'Weekday' if weekday else 'Weekend'} model)",
            xaxis_title="Departure time (30-min bins)",
            yaxis_title="Probability (stacked to 1)",
            height=420,
            margin=dict(l=20, r=20, t=60, b=20),
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.02, xanchor="left", x=0),
        )
        fig_area.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            range=[-0.5, 47.5],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
        )
        fig_area.update_yaxes(
            range=[0, 1],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
        )

        st.plotly_chart(fig_area, width="stretch")
        st.caption(
            "Each area shows how often the model believes the corridor is in a given latent regime at that time-of-day.")

        # ------------------------------------------------------------
        # Plot B: Baseline travel time + state mean curves
        # ------------------------------------------------------------
        st.markdown("### Baseline travel time and regime-conditioned means")

        fig_base = go.Figure()

        # baseline curve
        fig_base.add_trace(go.Scatter(
            x=tod_bin,
            y=base,
            mode="lines+markers",
            name="baseline mean",
            hovertemplate="bin=%{x} (%{customdata})<br>baseline=%{y:.1f} min<extra></extra>",
            customdata=tod_labels,
            line=dict(width=3),
        ))

        # state mean curves (baseline + residual mean)
        # (keep as lines; can be many traces but K<=6 in your UI so ok)
        for k in range(K_states):
            fig_base.add_trace(go.Scatter(
                x=tod_bin,
                y=state_mean_curves[:, k],
                mode="lines",
                name=f"state {k} mean",
                hovertemplate="bin=%{x} (%{customdata})<br>state-mean=%{y:.1f} min<extra></extra>",
                customdata=tod_labels,
                line=dict(dash="dash"),
                opacity=0.9,
            ))

        # selected bin marker
        fig_base.add_shape(
            type="line",
            x0=selected_bin, x1=selected_bin,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(width=2, dash="dot"),
        )

        fig_base.update_layout(
            template="plotly_white",
            title="Baseline mean curve and regime-conditioned mean travel times",
            xaxis_title="Departure time (30-min bins)",
            yaxis_title="Travel time (minutes)",
            height=420,
            margin=dict(l=20, r=20, t=60, b=20),
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.02, xanchor="left", x=0),
        )
        fig_base.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            range=[-0.5, 47.5],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
        )
        fig_base.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")

        st.plotly_chart(fig_base, width="stretch")
        st.caption(
            "The HMM is trained on residuals: residual = T - baseline(time). "
            "So each regime shifts the baseline up/down by its residual mean."
        )

        # ------------------------------------------------------------
        # Compact state parameter table (residual space)
        # ------------------------------------------------------------
        st.markdown("### State emission parameters (residual space)")

        # Sort by residual mean for interpretability
        order = np.argsort(mu_res)
        info = pd.DataFrame({
            "state": order,
            "residual_mean (min)": mu_res[order],
            "residual_std (min)": sig_res[order],
            "selected-bin weight": (hmm_w[order] if hmm_w is not None else np.nan),
        })
        st.dataframe(info, width='stretch')

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
