from __future__ import annotations
import numpy as np
import pandas as pd


def compute_station_lengths(df: pd.DataFrame) -> pd.Series:
    """
    Use a stable length per station. Median is robust to outliers/blanks.
    """
    L = df.groupby("Station")["StationLength"].median()
    return L


def travel_time_samples(df: pd.DataFrame, ordered_stations: list[int],
                        origin_station: int, dest_station: int,
                        min_coverage: float = 0.90) -> pd.Series:
    """
    Returns a Series indexed by Timestamp with travel time in minutes,
    computed as sum(length/speed) across stations between origin and dest (inclusive).
    """
    if origin_station not in ordered_stations or dest_station not in ordered_stations:
        raise ValueError("origin or dest not in ordered stations list")

    i = ordered_stations.index(origin_station)
    j = ordered_stations.index(dest_station)
    if i == j:
        raise ValueError("origin and destination are the same station")
    if i > j:
        i, j = j, i

    route_stations = ordered_stations[i:j+1]

    # Pivot speeds for those stations
    x = df[df["Station"].isin(route_stations)].copy()
    speed = x.pivot_table(index="Timestamp", columns="Station",
                          values="AvgSpeed", aggfunc="mean")

    # Station lengths (use median per station)
    L = compute_station_lengths(x).reindex(route_stations)

    # Ensure columns ordered
    speed = speed.reindex(columns=route_stations)

    # Coverage filter: require enough stations present at timestamp
    coverage = speed.notna().mean(axis=1)
    speed = speed[coverage >= min_coverage]

    # Compute time (hours) = sum_i L_i / v_i
    # Use broadcasting: L (station,) / speed (time, station)
    time_hours = (speed.values ** -1) * L.values  # elementwise L / v
    T_minutes = np.nansum(time_hours, axis=1) * 60.0

    return pd.Series(T_minutes, index=speed.index, name="TravelTimeMin")
