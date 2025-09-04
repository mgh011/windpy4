#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 08:02:21 2025

@author: mauro_ghirardelli
"""

from __future__ import annotations
import os
import pickle
from glob import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import json



#
#
# -------------- BLOCK I: MB Peak
#
#
#



#
#
# -------------- BLOCK II: UP Coherence
#
#
#

#
#
# -------------- BLOCK III: UW covariance low
#
#
#

#
#
# -------------- BLOCK IV: Inertial subrange
#
#
#



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-file, structured workflow for station QC filtering and plotting.

Layout (in-file modules):
- PRIMARY: pipeline entry points (call these from your main script)
- PRIMARY: plotting (public plotting API)
- HELPER:  I/O (file discovery + loading)
- HELPER:  QC logic (criteria per step)
- HELPER:  mask utils (compose/convert)
- PRIMARY: __main__ demo (optional CLI-style usage)

Extendability:
- Add future steps (step2, step3, ...) inside the QC section and call them
  from the pipeline orchestration while keeping helpers reusable.
"""




# =============================================================================
# PRIMARY — PIPELINE ENTRY POINTS
# =============================================================================
def process_station_qc_step1(
    base_path: str,
    station: str,
    qc_threshold: float = 0.05,
    *,
    plot: bool = True,
    plot_kwargs: Optional[dict] = None,
) -> List[Tuple[pd.Timestamp, int]]:
    """
    Stream over all daily pickle files for a station (one-by-one, memory-safe),
    build per-file QC masks (Step 1), convert to pairs, aggregate, and plot.

    Parameters
    ----------
    base_path : str
        Root folder that contains station subfolders (e.g., ".../data/").
    station : str
        Station name like "st1".
    qc_threshold : float, optional
        QC fraction threshold (0..1), default 0.05 (5%).
    plot : bool, optional
        If True, plot aggregated valid points per station.
    plot_kwargs : dict or None, optional
        Extra keyword args forwarded to `plot_pairs_by_station`.

    Returns
    -------
    list[(Timestamp, int)]
        Aggregated (time, height) pairs that passed Step 1 QC across all files.
    """
    file_paths = list_station_files(base_path, station, pattern="*.pkl")
    print(f"\n--- {station} ---")
    print(f"Found {len(file_paths)} daily files.")

    pairs_all: List[Tuple[pd.Timestamp, int]] = []

    for fp in file_paths:
        try:
            ds = load_daily_ds(fp)  # dict with keys like 'stats', 'spectra', ...
        except Exception as e:
            print(f"  ! Failed to load {Path(fp).name}: {e}")
            continue

        mask_qc = make_qc_mask_step1(ds, qc_threshold=qc_threshold)
        pairs = mask_to_pairs(mask_qc)
        pairs_all.extend(pairs)

        # Explicitly free references
        del ds, mask_qc

    if plot:
        plot_pairs_by_station(
            pairs_all,
            station_name=f"{station} (QC ≤ {qc_threshold:.2f})",
            **(plot_kwargs or {})
        )

    return pairs_all


# =============================================================================
# PRIMARY — PLOTTING (PUBLIC)
# =============================================================================
def plot_pairs_by_station(
    pairs: List[Tuple[pd.Timestamp, int]],
    station_name: str = "",
    *,
    marker_size: int = 6,
    alpha: float = 0.9,
    figsize: Optional[Tuple[int, int]] = None,
    jitter_minutes: float = 0.0,
    sort_days: str = "asc",
    show_legend: bool = True,
):
    """
    Visualize (time, height) points as scatter along a day index (y-axis)
    and time-of-day (x-axis).
    """
    if not pairs:
        print(f"{station_name}: no points to plot.")
        return

    df = pd.DataFrame(pairs, columns=["time", "height"])
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    df["tod_hours"] = (
        df["time"].dt.hour
        + df["time"].dt.minute / 60.0
        + df["time"].dt.second / 3600.0
    )

    # y coordinate by day
    days = np.sort(df["date"].unique())
    if sort_days == "desc":
        days = days[::-1]
    day_to_y = {d: i for i, d in enumerate(days)}
    df["y"] = df["date"].map(day_to_y)

    # x coordinate + small per-height offsets
    heights = sorted(df["height"].unique())
    if len(heights) == 1:
        offsets = {heights[0]: 0.0}
    else:
        base = 0.04  # ~2.4 minutes
        half = (len(heights) - 1) / 2
        offsets = {h: base * (i - half) for i, h in enumerate(heights)}
    df["x"] = df["tod_hours"] + df["height"].map(offsets)

    # optional jitter
    if jitter_minutes and jitter_minutes > 0:
        jitter = (np.random.rand(len(df)) - 0.5) * (2 * jitter_minutes) / 60.0
        df["x"] = df["x"] + jitter

    # colors per height from default cycle
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_map = {h: prop_cycle[i % max(1, len(prop_cycle))] for i, h in enumerate(heights)}
    df["color"] = df["height"].map(color_map)

    # figure size
    if figsize is None:
        figsize = (20, max(6, int(len(days) * 0.4)))

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    for h in heights:
        sub = df[df["height"] == h]
        ax.scatter(sub["x"], sub["y"], s=marker_size, alpha=alpha, c=sub["color"], label=f"h={h} m")

    ax.set_xlim(0, 24)
    ax.set_xticks(range(25))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(25)])
    ax.set_yticks(range(len(days)))
    ax.set_yticklabels([pd.to_datetime(d).strftime("%Y-%m-%d") for d in days])
    ax.set_xlabel("Time of day")

    n_points = len(df)
    ax.set_title(f"{station_name}: valid points ({n_points} across {len(days)} days)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    if show_legend and len(heights) > 1:
        ax.legend(title="Height", loc="upper right", frameon=False)

    fig.tight_layout()
    plt.show()


# =============================================================================
# HELPER — I/O
# =============================================================================
def list_station_files(base_path: str, station: str, pattern: str = "*.pkl") -> List[str]:
    """
    Return a sorted list of pickle files for a given station (under <base_path>/<station>/).
    """
    folder = os.path.join(base_path, station)
    return sorted(glob(os.path.join(folder, pattern)))


def load_daily_ds(pkl_path: str):
    """
    Load a daily pickle containing the station object:
    expected dict with keys like 'stats', 'spectra', 'coherence', 'autocorr', 'anisotropy_smooth'.
    """
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# HELPER — QC LOGIC (STEP 1 + scaffolding for future steps)
# =============================================================================
def select_qc_field(stats: xr.Dataset) -> Optional[str]:
    """
    Choose which QC field to use. Preference:
    1) 'QCnan' (fraction 0..1; lower is better)
    2) 'QC'    (fraction 0..1 or 0..100; will normalize if > 1)
    """
    if "QCnan" in stats.data_vars:
        return "QCnan"
    if "QC" in stats.data_vars:
        return "QC"
    return None


def _normalize_qc_fraction(qc: xr.DataArray) -> xr.DataArray:
    """
    Ensure QC is in [0,1]. If max > 1.0, interpret as percentage and divide by 100.
    """
    return qc / 100.0 if float(qc.max().values) > 1.0 else qc


def make_qc_mask_step1_from_stats(stats: xr.Dataset, qc_threshold: float = 0.05) -> xr.DataArray:
    """
    Build boolean mask (time, heights) using QC threshold if available.
    Fallback: non-NaN checks on basic stats and optional boolean flags.
    """
    mask = xr.DataArray(
        np.ones((stats.sizes["time"], stats.sizes["heights"]), dtype=bool),
        coords={"time": stats["time"], "heights": stats["heights"]},
        dims=("time", "heights"),
        name="mask_step1_qc",
    )

    qc_name = select_qc_field(stats)
    if qc_name is not None:
        qc = _normalize_qc_fraction(stats[qc_name])
        mask = mask & (qc <= qc_threshold) & (~xr.apply_ufunc(np.isnan, qc))
        return mask.astype(bool)

    # Fallback path (no QC field found)
    base_vars = [v for v in ("meanU", "uu", "vv", "ww") if v in stats.data_vars]
    if base_vars:
        base_ok = xr.ones_like(mask, dtype=bool)
        for v in base_vars:
            base_ok = base_ok & (~xr.apply_ufunc(np.isnan, stats[v]))
        mask = mask & base_ok

    for flag in ("statU", "statUW", "statWT"):
        if flag in stats.data_vars:
            mask = mask & (stats[flag].astype(bool))

    return mask.astype(bool)


def make_qc_mask_step1(ds, qc_threshold: float = 0.05) -> xr.DataArray:
    """
    Wrapper using the station object (dict with a 'stats' xr.Dataset).
    """
    if not isinstance(ds, dict) or "stats" not in ds:
        raise ValueError("Input must be a dict containing a 'stats' xarray.Dataset.")
    return make_qc_mask_step1_from_stats(ds["stats"], qc_threshold=qc_threshold)


# =============================================================================
# HELPER — MASK UTILS
# =============================================================================
def mask_to_pairs(mask_da: xr.DataArray) -> List[Tuple[pd.Timestamp, int]]:
    """
    Convert a (time, heights) boolean mask into a list of (Timestamp, height) pairs.
    """
    if not isinstance(mask_da, xr.DataArray):
        raise TypeError("mask_da must be an xarray.DataArray.")
    mask_da = mask_da.transpose("time", "heights")
    stacked = mask_da.stack(idx=("time", "heights"))
    idx_true = np.flatnonzero(stacked.values)
    times = pd.to_datetime(stacked["time"].values[idx_true])
    heights = stacked["heights"].values[idx_true]
    return list(zip(times, heights.tolist()))

# =========================
# =========================
# =========================
# Robust MB filter (global robust z across all stations)
# =========================
# =========================
# =========================

def robust_z(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Compute robust z-scores using median and MAD:
        z = (x - median) / (1.4826 * MAD)
    If MAD == 0, fall back to a very small value to avoid division by zero.

    Returns
    -------
    z : np.ndarray
        Robust z-scores.
    med : float
        Median of x.
    mad : float
        Median Absolute Deviation (unscaled).
    """
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = (1.4826 * mad) if mad > 0 else 1e-12
    z = (x - med) / scale
    return z, float(med), float(mad)


def gather_mb_rows_for_station(
    base_path: str,
    station: str,
    pairs_step1: list[tuple[pd.Timestamp, int]],
    MB_rel: bool = False,
) -> pd.DataFrame:
    """
    For a given station and its Step-1 valid pairs (time, height),
    stream over ALL daily files of the station, and extract MB = MB_peak / MB_fit
    only at those (time, height) coordinates.

    This avoids building any date index; we simply iterate over files and pick
    the pairs that belong to each file (based on time membership).

    Returns
    -------
    rows_df : pd.DataFrame with columns:
        ['station', 'time', 'height', 'MB_value']
    """
    file_paths = list_station_files(base_path, station, pattern="*.pkl")
    if not pairs_step1:
        return pd.DataFrame(columns=["station", "time", "height", "MB_value"])

    # Prepare a dict: for quick filtering per file, group pairs by time window
    # We don't know file boundaries, so we load each file and then select pairs inside.
    rows = []

    # Convert to numpy for speed
    pairs_np = [(np.datetime64(pd.to_datetime(t)), int(h)) for t, h in pairs_step1]

    for fp in file_paths:
        try:
            ds = load_daily_ds(fp)  # {'stats': xr.Dataset, ...}
        except Exception as e:
            print(f"  ! {station}: failed to load {Path(fp).name}: {e}")
            continue

        if not isinstance(ds, dict) or "stats" not in ds:
            del ds
            continue

        stats = ds["stats"]
        # Defensive checks
        if "MB_peak" not in stats.data_vars or "MB_fit" not in stats.data_vars:
            del ds
            continue


        MB = (stats["MB_peak"] / stats["MB_fit"]).astype(float)  # (time, heights)
        #MB = (stats["MB_peak"]).astype(float)  # (time, heights)
        times_file = pd.to_datetime(stats["time"].values)  # exact coordinates
        heights_file = stats["heights"].values

        # Build a fast membership structure for this file
        # (exact matching should work because Step-1 times came from these files)
        time_set = set(np.datetime64(t) for t in times_file)

        # Filter the pairs that belong to this file by time membership
        pairs_in_file = [(t, h) for (t, h) in pairs_np if t in time_set]
        if not pairs_in_file:
            del ds, MB
            continue

        # For each pair, extract MB value (exact select; fallback to nearest if needed)
        for (t, h) in pairs_in_file:
            try:
                mb_val = MB.sel(time=t, heights=h).item()
            except Exception:
                # fallback if tiny rounding differences exist
                mb_val = MB.sel(time=t, heights=h, method="nearest").item()
            rows.append({
                "station": station,
                "time": pd.to_datetime(t),
                "height": int(h),
                "MB_value": float(mb_val),
            })

        del ds, MB  # free memory

    if not rows:
        return pd.DataFrame(columns=["station", "time", "height", "MB_value"])

    rows_df = pd.DataFrame(rows).sort_values(["station", "time", "height"]).reset_index(drop=True)
    return rows_df


def filter_pairs_by_global_MB(
    mb_df_all: pd.DataFrame,
    tau_z: float = 3.0,
    direction: str = "above",
) -> tuple[pd.DataFrame, dict]:
    """
    Apply a global robust-z filter on MB values across ALL stations combined.

    Parameters
    ----------
    mb_df_all : DataFrame
        Must contain 'station', 'time', 'height', 'MB_value'.
    tau_z : float
        Robust z-score threshold (e.g., 3.0).
    direction : {"above","below","abs"}
        - "above": keep points with z >= tau_z (high-MB tail)
        - "below": keep points with z <= -tau_z (low-MB tail)
        - "abs":   keep points with |z| >= tau_z (two-sided outliers)

    Returns
    -------
    filtered_df : DataFrame
        Same columns as input + ['MB_z', 'MB_flag'] only for rows that pass the filter.
    stats : dict
        {'median': med, 'mad': mad} used for z-score.
    """
    if mb_df_all.empty:
        return mb_df_all.copy(), {"median": np.nan, "mad": np.nan}

    z, med, mad = robust_z(mb_df_all["MB_value"].values)
    mb_df = mb_df_all.copy()
    mb_df["MB_z"] = z

    if direction == "above":
        mb_df["MB_flag"] = mb_df["MB_z"] >= tau_z
    elif direction == "below":
        mb_df["MB_flag"] = mb_df["MB_z"] <= -tau_z
    elif direction == "abs":
        mb_df["MB_flag"] = np.abs(mb_df["MB_z"]) >= tau_z
    else:
        raise ValueError("direction must be 'above', 'below', or 'abs'.")

    filtered_df = mb_df.loc[mb_df["MB_flag"]].reset_index(drop=True)
    return filtered_df, {"median": med, "mad": mad}


def filter_MB_across_stations(
    base_path: str,
    pairs_by_station: dict[str, list[tuple[pd.Timestamp, int]]],
    *,
    tau_z: float = 3.0,
    direction: str = "above",
    plot: bool = True,
    plot_kwargs: Optional[dict] = None,
) -> dict[str, list[tuple[pd.Timestamp, int]]]:
    """
    Step 2 orchestrator:
      - For each station, load all files and collect MB values at Step-1 pairs.
      - Pool ALL stations' MB values to compute a GLOBAL robust z.
      - Filter by robust-z threshold.
      - Return filtered pairs per station and optionally plot them.

    Parameters
    ----------
    base_path : str
        Root folder with station subfolders.
    pairs_by_station : dict
        {'st1': [(time, h), ...], 'st2': [...], ...} from Step 1.
    tau_z : float
        Robust z threshold.
    direction : {"above","below","abs"}
        Which tail(s) to keep (see filter_pairs_by_global_MB).
    plot : bool
        Plot the filtered pairs per station.
    plot_kwargs : dict or None
        Extra args forwarded to plot_pairs_by_station.

    Returns
    -------
    filtered_pairs_by_station : dict[str, list[(Timestamp, int)]]
    """
    # 1) Gather MB rows for all stations
    all_rows = []
    for station, pairs in pairs_by_station.items():
        print(f"\n[Step 2] Gathering MB for {station} ({len(pairs)} pairs from Step 1)")
        df_station = gather_mb_rows_for_station(base_path, station, pairs)
        if not df_station.empty:
            all_rows.append(df_station)

    if not all_rows:
        print("No MB data collected in Step 2.")
        return {k: [] for k in pairs_by_station.keys()}

    mb_all = pd.concat(all_rows, ignore_index=True)

    # 2) Global robust-z filter
    filt_df, stats = filter_pairs_by_global_MB(mb_all, tau_z=tau_z, direction=direction)
    print(f"\n[Step 2] Global robust-z: median={stats['median']:.6g}, MAD={stats['mad']:.6g}")
    print(f"[Step 2] Kept {len(filt_df)} / {len(mb_all)} points (direction='{direction}', tau_z={tau_z})")

    # 3) Build filtered pairs per station
    filtered_pairs_by_station: dict[str, list[tuple[pd.Timestamp, int]]] = {}
    for station in pairs_by_station.keys():
        sub = filt_df.loc[filt_df["station"] == station]
        if sub.empty:
            filtered_pairs_by_station[station] = []
            if plot:
                plot_pairs_by_station([], station_name=f"{station} (Step 2 filtered)")
            continue

        pairs = list(zip(pd.to_datetime(sub["time"]).to_list(), sub["height"].astype(int).to_list()))
        filtered_pairs_by_station[station] = pairs

        if plot:
            plot_pairs_by_station(
                pairs,
                station_name=f"{station} (Step 2: MB robust-z {direction} ≥ {tau_z})",
                **(plot_kwargs or {})
            )

    return filtered_pairs_by_station

# ========================= # ========================= #
# ========================= # ========================= #
# ========================= # ========================= #
#     Coherence gate around the microbarom peak         #
# ========================= # ========================= #
# ========================= # ========================= #
# ========================= # ========================= #

def _sp_peak_in_band(f_spec, Spp, band) -> tuple[bool, Optional[float]]:
    """
    Return (has_peak, f_star) where f_star is argmax(Spp) within `band`.
    False if there are no finite samples in the band.
    """
    mask = (f_spec >= band[0]) & (f_spec <= band[1])
    if not mask.any():
        return False, None
    vals = np.asarray(Spp[mask], dtype=float)
    if vals.size == 0 or not np.any(np.isfinite(vals)):
        return False, None
    i_rel = int(np.nanargmax(vals))
    i_abs = np.where(mask)[0][i_rel]
    return True, float(f_spec[i_abs])

def _has_coh_peak(f_coh, coh_arr, f_star, win, thr=0.40, require_local=True) -> bool:
    """
    True if coherence series has a peak >= `thr` within ±`win` around f_star.
    If `require_local` is True, require a strict local maximum (greater than neighbors).
    """
    if coh_arr is None:
        return False
    wmask = (f_coh >= f_star - win) & (f_coh <= f_star + win)
    if np.count_nonzero(wmask) < 3:
        return False
    seg = np.asarray(coh_arr[wmask], dtype=float)
    if not np.any(np.isfinite(seg)):
        return False
    j = int(np.nanargmax(seg))
    peak = float(seg[j])
    if not np.isfinite(peak) or peak < thr:
        return False
    if require_local:
        if j == 0 or j == len(seg) - 1:
            return False
        if not (seg[j] > seg[j-1] and seg[j] > seg[j+1]):
            return False
    return True

def passes_filter3_simple(
    f_spec, Spp, f_coh, coh_up, coh_wp,
    band=(0.1, 0.4), win=0.03, thr=0.40, mode="any"
) -> bool:
    """
    Decide whether a (time, height) sample passes Filter 3:
      - f* = argmax(sp) within `band`
      - within ±`win` around f*, require a coherence peak >= `thr`
        according to `mode`:
          'any'  : u–p OR w–p
          'both' : u–p AND w–p
          'u_only': only u–p
          'w_only': only w–p
    """
    ok_sp, f_star = _sp_peak_in_band(f_spec, Spp, band)
    if not ok_sp:
        return False

    up_ok = _has_coh_peak(f_coh, coh_up, f_star, win, thr=thr, require_local=True)
    wp_ok = _has_coh_peak(f_coh, coh_wp, f_star, win, thr=thr, require_local=True) if coh_wp is not None else False

    if mode == "both":
        return up_ok and wp_ok
    if mode == "u_only":
        return up_ok
    if mode == "w_only":
        return wp_ok
    return up_ok or wp_ok  # 'any'

def _pairs_in_file_by_time(ds_like: xr.Dataset, pairs_input: list[tuple[pd.Timestamp, int]]) -> list[tuple[np.datetime64, int]]:
    """
    Select (t,h) pairs whose time exists in ds_like['time'].
    Returns times as np.datetime64 for exact matching.
    """
    if ds_like is None or "time" not in ds_like:
        return []
    time_set = set(np.datetime64(t) for t in pd.to_datetime(ds_like["time"].values))
    out: list[tuple[np.datetime64, int]] = []
    for t, h in pairs_input:
        t64 = np.datetime64(pd.to_datetime(t))
        if t64 in time_set:
            out.append((t64, int(h)))
    return out

def process_station_filter3(
    base_path: str,
    station: str,
    pairs_step2: list[tuple[pd.Timestamp, int]],
    *,
    band: tuple[float, float] = (0.1, 0.4),
    win: float = 0.03,
    thr: float = 0.40,
    mode: str = "any",
    plot: bool = True,
    plot_kwargs: Optional[dict] = None,
) -> list[tuple[pd.Timestamp, int]]:
    """
    Stream through 10-min pickles of `station` (pattern '*_10min.pkl') and keep only (time, height)
    pairs that satisfy Filter 3: sp-peak in `band` and coherence peak >= `thr` within ±`win`
    for the selected component(s) per `mode`.

    Returns
    -------
    list[(Timestamp, int)]
        Filtered (time, height) pairs for this station.
    """
    if not pairs_step2:
        if plot:
            plot_pairs_by_station([], station_name=f"{station} (Step 3 — no input)")
        return []

    file_paths = list_station_files(base_path, station, pattern="*_10min.pkl")
    kept: list[tuple[pd.Timestamp, int]] = []

    for fp in file_paths:
        try:
            ds = load_daily_ds(fp)  # {'spectra': xr.Dataset, 'coherence': xr.Dataset, ...}
        except Exception as e:
            print(f"  ! {station}: failed to load {Path(fp).name}: {e}")
            continue

        spectra = ds.get("spectra")
        coherence = ds.get("coherence")
        if spectra is None or coherence is None:
            del ds
            continue

        # restrict to pairs present in this file's time axis
        pairs_in_file = _pairs_in_file_by_time(spectra, pairs_step2)
        if not pairs_in_file:
            del ds, spectra, coherence
            continue

        f_spec = np.asarray(spectra["freq"].values, dtype=float)
        f_coh  = np.asarray(coherence["freq"].values, dtype=float)

        for t64, h in pairs_in_file:
            try:
                Spp    = spectra["sp"].sel(time=t64, heights=h, method="nearest").values
                coh_up = coherence["coh_up"].sel(time=t64, heights=h, method="nearest").values
                coh_wp = coherence["coh_wp"].sel(time=t64, heights=h, method="nearest").values if "coh_wp" in coherence else None
            except Exception:
                continue

            keep = passes_filter3_simple(
                f_spec=f_spec, Spp=Spp,
                f_coh=f_coh, coh_up=coh_up, coh_wp=coh_wp,
                band=band, win=win, thr=thr, mode=mode
            )
            if keep:
                kept.append((pd.Timestamp(t64), int(h)))

        del ds, spectra, coherence

    if plot:
        plot_pairs_by_station(
            kept,
            station_name=f"{station} (Step 3: coh≥{thr:.2f}, win±{win}, band {band[0]}–{band[1]} Hz, mode={mode})",
            **(plot_kwargs or {})
        )

    return kept

def process_coherence_across_stations(
    base_path: str,
    pairs_by_station: dict[str, list[tuple[pd.Timestamp, int]]],
    *,
    band: tuple[float, float] = (0.1, 0.4),
    win: float = 0.03,
    thr: float = 0.40,
    mode: str = "any",
    plot: bool = True,
    plot_kwargs: Optional[dict] = None,
) -> dict[str, list[tuple[pd.Timestamp, int]]]:
    """
    Apply Step 3 to stations st1..st6 and return {station: [(Timestamp, height), ...]}.
    Always return all six keys; empty lists if no matches.
    """
    stations = [f"st{i}" for i in range(1, 7)]
    out: dict[str, list[tuple[pd.Timestamp, int]]] = {}
    for st in stations:
        pairs_in = pairs_by_station.get(st, [])
        out[st] = process_station_filter3(
            base_path=base_path,
            station=st,
            pairs_step2=pairs_in,
            band=band, win=win, thr=thr, mode=mode,
            plot=plot,
            plot_kwargs=plot_kwargs,
        )
    return out

# ========================= # ========================= #
# ========================= # ========================= #
# ========================= # ========================= #
#          Inertial Subrange (Kaimal) gate              #
# ========================= # ========================= #
# ========================= # ========================= #
# ========================= # ========================= #


def _kaimal_f0_fm_from_intlenW_meanU(
    stats: xr.Dataset,
    t64: np.datetime64,
    h: int,
    *,
    Lw_var: str = "intlenW",
    U_var: str = "meanU",
    c0: float = 0.164/3.8,   # your choice
    cm: float = 0.164        # consistent with f0*4
) -> tuple[Optional[float], Optional[float]]:
    """
    Compute (f0, fm) at (t64,h) from Kaimal-style scaling:
        f0 = c0 * U / Lw
        fm = cm * U / Lw
    using integral length scale of W (intlenW) and meanU.

    Returns (f0, fm) or (None, None) if missing.
    """
    if (Lw_var not in stats.data_vars) or (U_var not in stats.data_vars):
        return None, None
    try:
        Lw = stats[Lw_var].sel(time=t64, heights=h, method="nearest").values
        U  = stats[U_var].sel(time=t64, heights=h, method="nearest").values
        Lw = float(Lw) if np.size(Lw) == 1 else float(np.nanmean(Lw))
        U  = float(U)  if np.size(U)  == 1 else float(np.nanmean(U))
        if not (np.isfinite(Lw) and Lw > 0 and np.isfinite(U) and U > 0):
            return None, None
        f0 = c0 * U / Lw
        fm = cm * U / Lw
        return float(f0), float(fm)
    except Exception:
        return None, None

def process_station_filter4_isr(
    base_path: str,
    station: str,
    pairs_step3: list[tuple[pd.Timestamp, int]],
    *,
    band: tuple[float, float] = (0.1, 0.4),
    keep_if: str = "after_fm",            # {"after_fm","before_fm"}
    require_fm_below_bandmin: bool = False,  # if True, require fm <= band[0]
    Lw_var: str = "intlenW",
    U_var: str = "meanU",
    c0: float = 0.164/3.8,
    cm: float = 0.164,
    plot: bool = True,
    plot_kwargs: Optional[dict] = None,
) -> list[tuple[pd.Timestamp, int]]:
    """
    ISR gate for a single station using Kaimal breaks from (intlenW, meanU).
    For each (t,h) that passed Step 3:
      - find f* (Spp peak in `band`)
      - compute (f0,fm) = (c0 * U/Lw, cm * U/Lw)
      - keep if condition holds:
           keep_if="after_fm"  -> f* >= fm   (microbarom peak lies in ISR)
           keep_if="before_fm" -> f* <  fm
      - optionally require fm <= band[0] (so the whole micro band is in/after ISR)
    """
    if not pairs_step3:
        if plot:
            plot_pairs_by_station([], station_name=f"{station} (Step 4 — no input)")
        return []

    file_paths = list_station_files(base_path, station, pattern="*_10min.pkl")
    kept: list[tuple[pd.Timestamp, int]] = []

    for fp in file_paths:
        try:
            d = load_daily_ds(fp)   # {'spectra':..., 'coherence':..., 'stats':...}
        except Exception as e:
            print(f"  ! {station}: failed to load {Path(fp).name}: {e}")
            continue

        spectra = d.get("spectra")
        stats   = d.get("stats")
        if spectra is None or stats is None or "time" not in spectra:
            continue

        # membership by time
        time_set = set(np.datetime64(tt) for tt in pd.to_datetime(spectra["time"].values))
        pairs_in_file = [(np.datetime64(pd.to_datetime(t)), int(h))
                         for (t, h) in pairs_step3 if np.datetime64(pd.to_datetime(t)) in time_set]
        if not pairs_in_file:
            continue

        f_spec = np.asarray(spectra["freq"].values, dtype=float)

        for t64, h in pairs_in_file:
            try:
                Spp = spectra["sp"].sel(time=t64, heights=h, method="nearest").values
            except Exception:
                continue

            ok_sp, f_star = _sp_peak_in_band(f_spec, Spp, band)
            if not ok_sp or f_star is None or not np.isfinite(f_star):
                continue

            f0, fm = _kaimal_f0_fm_from_intlenW_meanU(
                stats, t64, h, Lw_var=Lw_var, U_var=U_var, c0=c0, cm=cm
            )
            if fm is None or not np.isfinite(fm):
                continue

            if require_fm_below_bandmin and not (fm <= band[0]):
                continue

            keep = (f_star >= fm) if keep_if == "after_fm" else (f_star < fm)
            if keep:
                kept.append((pd.Timestamp(t64), int(h)))

    if plot:
        label = "f* ≥ fm (ISR)" if keep_if == "after_fm" else "f* < fm"
        if require_fm_below_bandmin:
            label += f", fm ≤ {band[0]:.2f} Hz"
        plot_pairs_by_station(
            kept,
            station_name=f"{station} (Step 4: {label})",
            **(plot_kwargs or {})
        )

    return kept

def process_ISR_across_stations(
    base_path: str,
    pairs_by_station: dict[str, list[tuple[pd.Timestamp, int]]],
    *,
    band: tuple[float, float] = (0.1, 0.4),
    keep_if: str = "after_fm",
    require_fm_below_bandmin: bool = False,
    Lw_var: str = "intlenW",
    U_var: str = "meanU",
    c0: float = 0.164/3.8,
    cm: float = 0.164,
    plot: bool = True,
    plot_kwargs: Optional[dict] = None,
) -> dict[str, list[tuple[pd.Timestamp, int]]]:
    """
    Apply ISR gate (Step 4) to st1..st6 and return {station: [(Timestamp, height), ...]}.
    Uses Kaimal breaks from (intlenW, meanU) with coefficients (c0, cm).
    """
    stations = [f"st{i}" for i in range(1, 7)]
    out: dict[str, list[tuple[pd.Timestamp, int]]] = {}
    for st in stations:
        pairs_in = pairs_by_station.get(st, [])
        out[st] = process_station_filter4_isr(
            base_path=base_path,
            station=st,
            pairs_step3=pairs_in,
            band=band,
            keep_if=keep_if,
            require_fm_below_bandmin=require_fm_below_bandmin,
            Lw_var=Lw_var,
            U_var=U_var,
            c0=c0,
            cm=cm,
            plot=plot,
            plot_kwargs=plot_kwargs,
        )
    return out


# =============================================================================
# RETURN MAPS / ANISOTROPY — density plots by start zone
# =============================================================================

# --- Geometry: Lumley triangle vertices (barycentric map) ---
_L_A = np.array([0.0, 0.0])                   # 2C corner
_L_C = np.array([0.5, np.sqrt(3)/2])          # 3C corner
_L_E = np.array([1.0, 0.0])                   # 1C corner
_L_H = float(np.sqrt(3)/2)

def _points_in_triangle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Vectorized point-in-triangle (A–C–E), inclusive borders.
    """
    x1, y1 = _L_A; x2, y2 = _L_C; x3, y3 = _L_E
    c1 = (x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)
    c2 = (x3 - x2)*(y - y2) - (y3 - y2)*(x - x2)
    c3 = (x1 - x3)*(y - y3) - (y1 - y3)*(x - x3)
    return ((c1 <= 0) & (c2 <= 0) & (c3 <= 0)) | ((c1 >= 0) & (c2 >= 0) & (c3 >= 0))

def _subtriangles(state: str, perc: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the two asymptotic sub-triangles for a given corner ('1c','2c','3c').
    `perc` (0..1) controls how deep into the main triangle they extend.
    """
    bary = np.array([0.5, np.sqrt(3)/6])  # centroid of base segment
    A, C, E = _L_A, _L_C, _L_E
    B = np.array([0.25, np.sqrt(3)/4])     # midpoint on A–C
    D = np.array([0.75, np.sqrt(3)/4])     # midpoint on C–E
    F = np.array([0.5,  0.0])              # midpoint on A–E

    if state == "1c":
        bary2 = np.array([1 - bary[0]*perc, bary[1]*perc])
        t1 = np.array([E, np.array([1 - B[0]*perc, D[1]*perc]), bary2])
        t2 = np.array([E, F*(2 - perc), bary2])
    elif state == "2c":
        bary2 = bary * perc
        t1 = np.array([A, B * perc, bary2])
        t2 = np.array([A, F * perc, bary2])
    elif state == "3c":
        bary2 = np.array([bary[0], C[1] - (C[1] - bary[1]) * perc])
        t1 = np.array([C, B * (2 - perc), bary2])
        t2 = np.array([C, np.array([1 - B[0]*(2 - perc), D[1]*(2 - perc)]), bary2])
    else:
        raise ValueError("state must be '1c','2c','3c'")
    return t1, t2

def _inside_strict(pt: Tuple[float,float], tri: np.ndarray) -> bool:
    """
    Strict (non-border) point-in-triangle for classification.
    """
    x, y = pt
    (x1,y1),(x2,y2),(x3,y3) = tri
    c1 = (x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)
    c2 = (x3 - x2)*(y - y2) - (y3 - y2)*(x - x2)
    c3 = (x1 - x3)*(y - y3) - (y1 - y3)*(x - x3)
    return (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0)

def _start_zone(x0: float, y0: float, perc: float = 0.7, h_ratio: float = 0.2) -> Optional[str]:
    """
    Classify the starting point (first frequency) into:
      '1c', '2c', or '2c_axi' (bottom rectangular belt).
    Returns None if it falls in none of these zones.
    """
    t1, t2 = _subtriangles('1c', perc)
    if _inside_strict((x0,y0), t1) or _inside_strict((x0,y0), t2):
        return '1c'
    t1, t2 = _subtriangles('2c', perc)
    if _inside_strict((x0,y0), t1) or _inside_strict((x0,y0), t2):
        return '2c'
    if (0.0 <= x0 <= 1.0) and (0.0 <= y0 <= h_ratio*_L_H):
        return '2c_axi'
    return None

def plot_barycentric_triangle(ax: plt.Axes, *, perc: float = 0.7, fill_zones: bool = True, show_labels: bool = True) -> None:
    """
    Draw the Lumley (barycentric) triangle and optionally shade the three start zones.
    """
    ax.plot([_L_A[0], _L_C[0], _L_E[0], _L_A[0]],
            [_L_A[1], _L_C[1], _L_E[1], _L_A[1]], color="k", lw=1)
    ax.scatter(9/22, 9/22  * 3**0.5, color="red", s=30)

    if fill_zones:
        # 1C (rose), 2C (light blue), 3C (light green)
        for st, color in (("1c", "#ffd4d4"), ("2c", "#d4e8ff"), ("3c", "#d7f7d7")):
            t1, t2 = _subtriangles(st, perc)
            ax.fill(*zip(*t1), color=color, alpha=0.35, lw=0)
            ax.fill(*zip(*t2), color=color, alpha=0.35, lw=0)

        # bottom belt (axi) in grey
        ax.fill([0,1,1,0], [0,0,_L_H*0.2,_L_H*0.2], color="#e9e9e9", alpha=0.35, lw=0)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, _L_H+0.02)
    ax.set_xticks([]); ax.set_yticks([])
    if show_labels:
        ax.text(_L_E[0], _L_E[1]-0.03, "1C", ha="center", va="top")
        ax.text(_L_A[0], _L_A[1]-0.03, "2C", ha="center", va="top")
        ax.text(_L_C[0], _L_C[1]+0.03, "3C", ha="center", va="bottom")


# ---------------- core collectors ----------------

def gather_anisotropy_tracks_for_station(
    base_path: str | Path,
    station: str,
    pairs: List[Tuple[pd.Timestamp, int]],
    *,
    fmax: float = 5.0,
    require_inside: bool = True,
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    Collect anisotropy tracks (xb, yb vs freq_cutoff) for the given (time,height)
    pairs of a station. Tracks are grouped by the START zone of the path:
    '1c', '2c', '2c_axi'. Optionally keep only tracks fully inside the triangle.

    Returns
    -------
    dict
      {
        '1c':     {'X': [x_arr,...], 'Y': [y_arr,...], 'F': [f_arr,...]},
        '2c':     {'X': [...],       'Y': [...],       'F': [...]},
        '2c_axi': {'X': [...],       'Y': [...],       'F': [...]}
      }
    """
    buckets = {
        '1c': {'X': [], 'Y': [], 'F': []},
        '2c': {'X': [], 'Y': [], 'F': []},
        '2c_axi': {'X': [], 'Y': [], 'F': []},
    }

    if not pairs:
        return buckets

    file_paths = list_station_files(base_path, station, pattern="*_10min.pkl")

    # per-file membership by time
    for fp in file_paths:
        try:
            d = load_daily_ds(fp)
        except Exception:
            continue

        ani = d.get("anisotropy_smooth")
        if ani is None or "time" not in ani or "heights" not in ani:
            continue

        times = ani["time"].values.astype("datetime64[ns]")
        heights = ani["heights"].values.astype(int)
        time_set = set(times)

        # keep only pairs present in this file
        pairs_here = []
        for t, h in pairs:
            t64 = np.datetime64(pd.to_datetime(t), "ns")
            if t64 in time_set and int(h) in heights:
                pairs_here.append((t64, int(h)))
        if not pairs_here:
            continue

        for t64, h in pairs_here:
            try:
                ds_sel = ani.sel(time=t64, heights=h, method="nearest")
                xb = np.asarray(ds_sel["xb"].values)
                yb = np.asarray(ds_sel["yb"].values)
                f  = np.asarray(ds_sel["freq_cutoff"].values, dtype=float)
            except Exception:
                continue

            if xb.size == 0 or yb.size == 0 or f.size == 0:
                continue

            # valid & <= fmax
            m = np.isfinite(f) & (f <= float(fmax))
            if not np.any(m):
                continue

            x_sub = xb[m]; y_sub = yb[m]; f_sub = f[m]

            # sort by frequency (robust, ensures monotonic grid downstream)
            order = np.argsort(f_sub)
            f_sub = f_sub[order]; x_sub = x_sub[order]; y_sub = y_sub[order]

            if require_inside and not np.all(_points_in_triangle(x_sub, y_sub)):
                continue

            zone = _start_zone(float(x_sub[0]), float(y_sub[0]))
            if zone is None:
                continue

            buckets[zone]['X'].append(x_sub)
            buckets[zone]['Y'].append(y_sub)
            buckets[zone]['F'].append(f_sub)

    return buckets


def plot_return_map_density_for_station(
    base_path: str | Path,
    station: str,
    pairs: List[Tuple[pd.Timestamp, int]],
    *,
    fmax: float = 5.0,
    perc: float = 0.7,
    h_ratio: float = 0.2,
    require_inside: bool = True,
    gridsize: int = 70,
    cmap: str = "Greys",
) -> None:
    """
    For a single station, draw three side-by-side hexbin densities on the Lumley
    triangle for tracks grouped by START zone: '1c', '2c', '2c_axi'.
    """
    buckets = gather_anisotropy_tracks_for_station(
        base_path, station, pairs, fmax=fmax, require_inside=require_inside
    )

    titles = {'1c': 'Start: 1C (lower-right)',
              '2c': 'Start: 2C (lower-left)',
              '2c_axi': f'Start: low belt (h≤{h_ratio}·H)'}
    order = ['1c', '2c', '2c_axi']

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    any_plotted = False
    last_hb = None

    for ax, key in zip(axs, order):
        plot_barycentric_triangle(ax=ax, perc=perc, fill_zones=True)
        if buckets[key]['X']:
            all_x = np.concatenate(buckets[key]['X'])
            all_y = np.concatenate(buckets[key]['Y'])
            hb = ax.hexbin(all_x, all_y, gridsize=gridsize, cmap=cmap,
                           bins='log', mincnt=1, linewidths=0.0)
            ax.set_title(f"{station} — {titles[key]}  |  n={len(buckets[key]['X'])}")
            last_hb = hb
            any_plotted = True
        else:
            ax.set_title(f"{station} — {titles[key]}  |  n=0")

    if not any_plotted:
        print(f"{station}: no valid tracks with current criteria.")
        plt.close(fig)
        return

    if last_hb is not None:
        cbar = fig.colorbar(last_hb, ax=axs, shrink=0.9)
        cbar.set_label("log(count)")

    plt.show()

def plot_return_maps_all_stations(
    base_path: str | Path,
    pairs_by_station: Dict[str, List[Tuple[pd.Timestamp, int]]],
    *,
    fmax: float = 5.0,
    perc: float = 0.7,
    h_ratio: float = 0.2,
    require_inside: bool = True,
    gridsize: int = 70,
    cmap: str = "Greys",
) -> None:
    """
    Loop over stations ('st1'..'st6') and produce one 3-panel density figure per station.
    """
    stations = [f"st{i}" for i in range(1, 7)]
    for st in stations:
        pairs = pairs_by_station.get(st, [])
        plot_return_map_density_for_station(
            base_path=base_path,
            station=st,
            pairs=pairs,
            fmax=fmax,
            perc=perc,
            h_ratio=h_ratio,
            require_inside=require_inside,
            gridsize=gridsize,
            cmap=cmap,
        )




def _compute_median_track(
    Xs: List[np.ndarray], Ys: List[np.ndarray], Fs: List[np.ndarray], *,
    fmax: float, grid_choice: str = "mode_len"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[None, None, None]:
    """
    Build a common frequency grid and compute the median track:
        x_med(f) = median_i x_i(f),  y_med(f) = median_i y_i(f).
    Each track (Xi,Yi) is linearly interpolated on the common grid.
    Grid strategy:
      - "mode_len": use the frequency vector of the most common length among tracks.
    Returns (f_ref, x_med, y_med); (None, None, None) if not enough data.
    """
    if not Xs:
        return None, None, None

    # choose a reference grid by most-common length
    lengths = [len(f) for f in Fs]
    mode_len = max(set(lengths), key=lengths.count)
    idx_ref = next(i for i, L in enumerate(lengths) if L == mode_len)
    f_ref = np.asarray(Fs[idx_ref], float)
    f_ref = f_ref[np.isfinite(f_ref) & (f_ref <= float(fmax))]
    if f_ref.size < 2:
        return None, None, None

    Xmat = np.full((len(Xs), f_ref.size), np.nan)
    Ymat = np.full((len(Ys), f_ref.size), np.nan)

    for i, (x, y, f) in enumerate(zip(Xs, Ys, Fs)):
        f = np.asarray(f, float)
        m = np.isfinite(f) & (f <= float(fmax))
        if np.count_nonzero(m) < 2:
            continue
        f_i = f[m]; x_i = np.asarray(x, float)[m]; y_i = np.asarray(y, float)[m]
        o = np.argsort(f_i); f_i, x_i, y_i = f_i[o], x_i[o], y_i[o]
        valid = (f_ref >= f_i[0]) & (f_ref <= f_i[-1])
        if np.count_nonzero(valid) == 0:
            continue
        Xmat[i, valid] = np.interp(f_ref[valid], f_i, x_i)
        Ymat[i, valid] = np.interp(f_ref[valid], f_i, y_i)

    if np.all(np.isnan(Xmat)) or np.all(np.isnan(Ymat)):
        return None, None, None

    x_med = np.nanmedian(Xmat, axis=0)
    y_med = np.nanmedian(Ymat, axis=0)
    return f_ref, x_med, y_med


from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

def _plot_gradient_path(ax, x, y, f, *, cmap="viridis", lw=2.0, zorder=2, show_cbar=False):
    """
    Plot a polyline (x,y) colored by the parameter f with a colormap.
    Returns the LineCollection (and optional colorbar if requested).
    """
    x = np.asarray(x, float); y = np.asarray(y, float); f = np.asarray(f, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(f)
    x, y, f = x[ok], y[ok], f[ok]
    if x.size < 2:
        return None, None

    pts = np.column_stack([x, y])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)

    norm = Normalize(vmin=f.min(), vmax=f.max())
    lc = LineCollection(segs, cmap=cmap, norm=norm)
    # use segment mid-frequency to color each segment
    f_mid = 0.5 * (f[:-1] + f[1:])
    lc.set_array(f_mid)
    lc.set_linewidth(lw)
    lc.set_zorder(zorder)
    ax.add_collection(lc)

    cbar = None
    if show_cbar:
        cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Frequency [Hz]")

    return lc, cbar


def _overlay_micro_segment(ax, x, y, f, micro_band, *, color="red", lw=3.0, zorder=3):
    """
    Draw only the segments whose BOTH endpoints fall inside micro_band.
    """
    x = np.asarray(x, float); y = np.asarray(y, float); f = np.asarray(f, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(f)
    x, y, f = x[ok], y[ok], f[ok]
    if x.size < 2:
        return

    pts = np.column_stack([x, y])
    # segments to draw = indices i where (f[i], f[i+1]) both in band
    m = (f >= micro_band[0]) & (f <= micro_band[1])
    seg_idx = np.where(m[:-1] & m[1:])[0]
    if seg_idx.size == 0:
        return
    segs = np.stack([pts[seg_idx], pts[seg_idx + 1]], axis=1)
    lc = LineCollection(segs, colors=color, linewidths=lw, zorder=zorder)
    ax.add_collection(lc)


def plot_return_map_density_with_median(
    base_path: str | Path,
    station: str,
    pairs: List[Tuple[pd.Timestamp, int]],
    *,
    fmax: float = 5.0,
    perc: float = 0.7,
    h_ratio: float = 0.2,
    require_inside: bool = True,
    gridsize: int = 70,
    cmap_bg: str = "Greys",
    micro_band: Tuple[float, float] = (0.1, 0.4),
    show_hexbin: bool = True,
    show_cbar: bool = False,        # set True per mostrare la colorbar delle frequenze
    cmap_curve: str = "viridis",    # curva mediana: scuro→chiaro (basse→alte f)
    lw_curve: float = 2.0,
    lw_micro: float = 3.0,
) -> None:
    """
    Hexbin density + median track per start-zone. The median track is drawn with a
    frequency-colored gradient (viridis by default), and the microbarom portion is
    over-plotted in solid red.

    - Background: optional hexbin of all points in the bucket.
    - Curve: median(x(f)), median(y(f)) over a common frequency grid.
             Colored by frequency to show direction (low f = scuro → high f = chiaro).
    - Micro band: the same median curve, limited to micro_band, in red.
    """
    buckets = gather_anisotropy_tracks_for_station(
        base_path, station, pairs, fmax=fmax, require_inside=require_inside
    )

    titles = {'1c': 'Start: 1C (lower-right)',
              '2c': 'Start: 2C (lower-left)',
              '2c_axi': f'Start: low belt (h≤{h_ratio}·H)'}
    order = ['1c', '2c', '2c_axi']

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    any_plotted = False
    last_hb = None

    for ax, key in zip(axs, order):
        plot_barycentric_triangle(ax=ax, perc=perc, fill_zones=True)

        B = buckets.get(key, {'X': [], 'Y': [], 'F': []})
        Xs, Ys, Fs = B.get('X', []), B.get('Y', []), B.get('F', [])
        if not Xs:
            ax.set_title(f"{station} — {titles[key]}  |  n=0")
            continue

        if show_hexbin:
            all_x = np.concatenate(Xs); all_y = np.concatenate(Ys)
            hb = ax.hexbin(all_x, all_y, gridsize=gridsize, cmap=cmap_bg,
                           bins='log', mincnt=1, linewidths=0.0)
            last_hb = hb

        # median track on a common frequency grid
        f_ref, x_med, y_med = _compute_median_track(Xs, Ys, Fs, fmax=fmax)
        if f_ref is None:
            ax.set_title(f"{station} — {titles[key]}  |  n={len(Xs)} (no median)")
            continue

        ok = np.isfinite(f_ref) & np.isfinite(x_med) & np.isfinite(y_med)
        f_ref, x_med, y_med = f_ref[ok], x_med[ok], y_med[ok]
        if f_ref.size < 2:
            ax.set_title(f"{station} — {titles[key]}  |  n={len(Xs)} (short)")
            continue

        # 1) gradient curve by frequency (direction cue)
        _plot_gradient_path(ax, x_med, y_med, f_ref,
                            cmap=cmap_curve, lw=lw_curve, zorder=3, show_cbar=show_cbar)

        # 2) micro-band segment in red
        _overlay_micro_segment(ax, x_med, y_med, f_ref, micro_band,
                               color='red', lw=lw_micro, zorder=4)

    
        ax.set_title(f"{station} — {titles[key]}  |  n={len(Xs)}")
        any_plotted = True

    if not any_plotted:
        print(f"{station}: no valid tracks with current criteria.")
        plt.close(fig); return

    if show_hexbin and last_hb is not None:
        cbar = fig.colorbar(last_hb, ax=axs, shrink=0.9)
        cbar.set_label("log(count)")

    plt.show()



def plot_return_map_density_with_median_cumulative(
    base_path: str | Path,
    pairs_by_station: Dict[str, List[Tuple[pd.Timestamp, int]]],
    *,
    fmax: float = 5.0,
    perc: float = 0.7,
    h_ratio: float = 0.2,     # kept for title consistency
    require_inside: bool = True,
    gridsize: int = 70,
    cmap_bg: str = "Greys",
    micro_band: Tuple[float, float] = (0.1, 0.4),
    show_hexbin: bool = True,
    show_cbar: bool = False,
    cmap_curve: str = "viridis",
    lw_curve: float = 2.0,
    lw_micro: float = 3.0,
) -> None:
    """
    CUMULATIVE return-map view across ALL stations.
    Everything is overlaid on the same three panels (1C, 2C, 2C_axi).

    Steps
    -----
    1) For each station, gather barycentric tracks (X,Y,F) for the given pairs.
    2) Aggregate all stations together per panel-key ('1c','2c','2c_axi').
    3) Plot a single hexbin background (optional) using all points from all stations.
    4) Compute a single global median track per panel over a common frequency grid.
    5) Overlay the micro-band segment of the median in solid red.

    Notes
    -----
    - Requires the helpers:
        gather_anisotropy_tracks_for_station(...)
        _compute_median_track(...)
        _plot_gradient_path(...)
        _overlay_micro_segment(...)
        plot_barycentric_triangle(...)
    - Median is computed across ALL tracks pooled together for that panel.
    """
    # 1) gather and 2) aggregate across stations
    agg = {k: {"X": [], "Y": [], "F": []} for k in ["1c", "2c", "2c_axi"]}

    total_tracks = {k: 0 for k in agg.keys()}
    for st, pairs in pairs_by_station.items():
        if not pairs:
            continue
        buckets = gather_anisotropy_tracks_for_station(
            base_path, st, pairs, fmax=fmax, require_inside=require_inside
        )
        for key in agg.keys():
            Xs, Ys, Fs = buckets.get(key, {}).get("X", []), buckets.get(key, {}).get("Y", []), buckets.get(key, {}).get("F", [])
            if Xs:
                agg[key]["X"].extend(Xs)
                agg[key]["Y"].extend(Ys)
                agg[key]["F"].extend(Fs)
                total_tracks[key] += len(Xs)

    titles = {
        "1c": "Start: 1C (lower-right)",
        "2c": "Start: 2C (lower-left)",
        "2c_axi": f"Start: low belt (h≤{h_ratio}·H)"
    }
    order = ["1c", "2c", "2c_axi"]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    last_hb = None
    anything = False

    for ax, key in zip(axs, order):
        plot_barycentric_triangle(ax=ax, perc=perc, fill_zones=True)

        Xs, Ys, Fs = agg[key]["X"], agg[key]["Y"], agg[key]["F"]
        n_tracks = total_tracks[key]
        if n_tracks == 0:
            ax.set_title(f"{titles[key]}  |  n=0")
            continue

        # 3) global hexbin background
        if show_hexbin:
            all_x = np.concatenate(Xs)
            all_y = np.concatenate(Ys)
            hb = ax.hexbin(
                all_x, all_y,
                gridsize=gridsize, cmap=cmap_bg,
                bins="log", mincnt=1, linewidths=0.0
            )
            last_hb = hb

        # 4) global median track on a common frequency grid
        f_ref, x_med, y_med = _compute_median_track(Xs, Ys, Fs, fmax=fmax)
        if f_ref is not None:
            ok = np.isfinite(f_ref) & np.isfinite(x_med) & np.isfinite(y_med)
            f_ref, x_med, y_med = f_ref[ok], x_med[ok], y_med[ok]
        if (f_ref is None) or (f_ref.size < 2):
            ax.set_title(f"{titles[key]}  |  n={n_tracks} (no median)")
            continue

        # gradient median curve (frequency-colored)
        _plot_gradient_path(
            ax, x_med, y_med, f_ref,
            cmap=cmap_curve, lw=lw_curve, zorder=3, show_cbar=False
        )

        # 5) microband segment in red
        _overlay_micro_segment(
            ax, x_med, y_med, f_ref, micro_band,
            color="red", lw=lw_micro, zorder=4
        )

        ax.set_title(f"{titles[key]}  |  n={n_tracks}")
        anything = True

    if not anything:
        print("No valid tracks to display with current criteria.")
        plt.close(fig)
        return

    if show_hexbin and show_cbar and last_hb is not None:
        cbar = fig.colorbar(last_hb, ax=axs, shrink=0.9)
        cbar.set_label("log(count)")

    plt.show()


# =========================
# Random spectra plots from Step-2 pairs (global sampler)
# =========================
from typing import Sequence

def _date_from_path(fp: str) -> Optional[str]:
    """Extract 'YYYY-MM-DD' from filenames like 'YYYY-MM-DD_*.pkl'."""
    try:
        return Path(fp).name.split("_")[0]
    except Exception:
        return None

def _build_date_map(files: Sequence[str]) -> Dict[str, str]:
    """Map date string -> file path for quick lookup."""
    d2fp = {}
    for fp in files:
        d = _date_from_path(fp)
        if d:
            d2fp[d] = fp
    return d2fp

def _find_file_for_time(station: str, t: pd.Timestamp,
                        files_by_station: Dict[str, List[str]],
                        date_map_by_station: Dict[str, Dict[str, str]]) -> Optional[str]:
    """Fast path by date map, then slow fallback by scanning."""
    day = pd.to_datetime(t).strftime("%Y-%m-%d")
    fp = date_map_by_station.get(station, {}).get(day)
    if fp and os.path.exists(fp):
        return fp
    # Fallback: scan files to find one containing this timestamp
    for cand in files_by_station.get(station, []):
        try:
            with open(cand, "rb") as f:
                d = pickle.load(f)
            times = pd.to_datetime(d["spectra"]["time"].values)
            if (times == pd.to_datetime(t)).any():
                return cand
        except Exception:
            continue
    return None

def _pick_random_pairs_across_stations(pairs_by_station: Dict[str, List[Tuple[pd.Timestamp, int]]],
                                       k: int, seed: int = 11) -> List[Tuple[str, pd.Timestamp, int]]:
    """Flatten all Step-2 pairs and pick k random (station, time, height)."""
    all_pairs = []
    for st, pairs in pairs_by_station.items():
        for (t, h) in pairs:
            all_pairs.append((st, pd.to_datetime(t), int(h)))
    if not all_pairs:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_pairs), size=min(k, len(all_pairs)), replace=False)
    return [all_pairs[i] for i in idx]

def plot_random_pressure_spectra(
    pairs_by_station: Dict[str, List[Tuple[pd.Timestamp, int]]],
    base_path: str,
    *,
    k: int = 6,
    seed: int = 11,
    fband: Tuple[float, float] = (0.1, 0.4),
) -> None:
    """
    Randomly plot pressure spectra across ALL stations for pairs that passed Step 2.

    Parameters
    ----------
    pairs_step2_by_station : dict
        {'st1': [(time, h), ...], ...} — pairs retained after Step 2.
    base_path : str
        Root folder containing station subfolders (base_path/st1/*.pkl, ...).
    k : int
        Number of examples to plot.
    seed : int
        RNG seed for reproducibility.
    fband : (float, float)
        Frequency band to highlight (e.g., microbarom band).

    Returns
    -------
    None (shows a matplotlib figure)
    """
    # Build file lists per station
    files_by_station = {
        st: sorted(glob(os.path.join(base_path, st, "*.pkl")))
        for st in pairs_by_station.keys()
    }
    # Build per-station date maps for fast file lookup
    date_map_by_station = {st: _build_date_map(fps) for st, fps in files_by_station.items()}

    # Randomly select pairs
    picks = _pick_random_pairs_across_stations(pairs_by_station, k=k, seed=seed)
    if not picks:
        print("No Step-2 pairs available to plot.")
        return

    ncols = 3
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    drawn = 0
    for ax, (station, t, h) in zip(axes.ravel(), picks):
        fp = _find_file_for_time(station, t, files_by_station, date_map_by_station)
        if fp is None:
            ax.set_visible(False)
            continue

        try:
            with open(fp, "rb") as f:
                d = pickle.load(f)
            spec: xr.Dataset = d["spectra"]
            fHz = spec["freq"].values
            Spp = spec["sp"].sel(time=np.datetime64(t), heights=int(h), method="nearest").values
        except Exception as e:
            print(f"[{station}] failed to read spectrum at {t} h={h} from {Path(fp).name}: {e}")
            ax.set_visible(False)
            continue

        ok = (fHz > 0) & (Spp > 0) & np.isfinite(Spp)
        if not np.any(ok):
            ax.set_visible(False)
            continue
        f = fHz[ok]; P = Spp[ok]

        ax.loglog(f, f * P, lw=1.5)
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel("f * Spp")
        day_str = pd.to_datetime(t).strftime("%Y-%m-%d")
        ax.set_title(f"{station} | {day_str}  t={pd.to_datetime(t).strftime('%H:%M')}  h={h}")

        fmin, fmax = fband
        ax.axvspan(fmin, fmax, alpha=0.15)
        ax.set_xlim(left=max(f.min(), 1e-3), right=min(f.max(), 10))

        drawn += 1

    for ax in axes.ravel()[drawn:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()




def plot_random_coherence(
    pairs_by_station: Dict[str, List[Tuple[pd.Timestamp, int]]],
    base_path: str,
    *,
    k: int = 6,
    seed: int = 11,
    fband: Tuple[float, float] = (0.1, 0.4),
) -> None:
    """
    Randomly pick K (station, time, height) examples from Step-2 pairs and,
    for EACH pick, plot a 3-panel figure stacked vertically:
      top:   f * Spp (pressure spectrum, pre-multiplied)
      middle:coh_up
      bottom:coh_uw  (if available; otherwise the figure has 2 panels)

    Notes
    -----
    - Reuses the same helpers as your original function:
      `_build_date_map`, `_pick_random_pairs_across_stations`, `_find_file_for_time`.
    - Frequency axis is shared (linear) across panels; the band `fband` is shaded.
    - The Spp peak f* within `fband` is marked, if detectable.

    Parameters
    ----------
    pairs_step2_by_station : dict
        {'st1': [(time, h), ...], ...} — pairs retained after Step 2.
    base_path : str
        Root folder containing station subfolders (base_path/st1/*.pkl, ...).
    k : int
        Number of random examples (thus, number of figures) to plot.
    seed : int
        RNG seed for reproducibility.
    fband : (float, float)
        Frequency band to highlight (e.g., microbarom band).
    """
    # Build file lists per station (same structure)
    files_by_station = {
        st: sorted(glob(os.path.join(base_path, st, "*.pkl")))
        for st in pairs_by_station.keys()
    }
    # Per-station date maps for fast lookup (same helper)
    date_map_by_station = {st: _build_date_map(fps) for st, fps in files_by_station.items()}

    # Randomly select pairs (same helper)
    picks = _pick_random_pairs_across_stations(pairs_by_station, k=k, seed=seed)
    if not picks:
        print("No Step-2 pairs available to plot.")
        return

    for station, t, h in picks:
        fp = _find_file_for_time(station, t, files_by_station, date_map_by_station)
        if fp is None:
            print(f"[{station}] No file found for {t} h={h}")
            continue

        try:
            with open(fp, "rb") as f:
                d = pickle.load(f)
            spec: xr.Dataset = d["spectra"]
            coh: xr.Dataset = d["coherence"]
        except Exception as e:
            print(f"[{station}] failed to load {Path(fp).name}: {e}")
            continue

        # Extract series at (t,h)
        t64 = np.datetime64(pd.to_datetime(t))
        try:
            f_spec = spec["freq"].values
            Spp    = spec["sp"].sel(time=t64, heights=int(h), method="nearest").values

            f_coh  = coh["freq"].values
            coh_up = coh["coh_up"].sel(time=t64, heights=int(h), method="nearest").values
            coh_uw = coh["coh_uw"].sel(time=t64, heights=int(h), method="nearest").values if "coh_uw" in coh else None
        except Exception as e:
            print(f"[{station}] selection error at {t} h={h} in {Path(fp).name}: {e}")
            continue

        # Compute f* (max of Spp in band)
        fmin, fmax = fband
        band_mask = (f_spec >= fmin) & (f_spec <= fmax)
        f_star = None
        if band_mask.any():
            vals = np.asarray(Spp[band_mask], dtype=float)
            if vals.size and np.any(np.isfinite(vals)):
                i_rel = int(np.nanargmax(vals))
                i_abs = np.where(band_mask)[0][i_rel]
                f_star = float(f_spec[i_abs])

        # Shared x-limits (overlap to keep axes aligned)
        x_min = max(np.nanmin(f_spec), np.nanmin(f_coh))
        x_max = min(np.nanmax(f_spec), np.nanmax(f_coh))

        # How many rows? 3 if coh_uw exists, else 2
        nrows = 3 if coh_uw is not None else 2
        fig, axes = plt.subplots(nrows, 1, figsize=(8, 9 if nrows == 3 else 7), sharex=True)

        # 1) Pressure spectrum (pre-multiplied)
        ax = axes[0]
        ok = np.isfinite(f_spec) & np.isfinite(Spp)
        f_ok = f_spec[ok]; P_ok = Spp[ok]
        ax.loglog(f_ok, f_ok * P_ok, lw=1.5)
        ax.axvspan(fmin, fmax, alpha=0.15, color="grey")
        if f_star is not None:
            ax.axvline(f_star, linestyle="--", lw=1)
        ax.set_xlim(x_min, x_max)
        ax.set_ylabel("f · Spp")
        day_str = pd.to_datetime(t).strftime("%Y-%m-%d")
        ax.set_title(f"{station} | {day_str}  t={pd.to_datetime(t).strftime('%H:%M')}  h={h}  |  {Path(fp).name}")

        # 2) Coherence u–p
        ax = axes[1]
        ok = np.isfinite(f_coh) & np.isfinite(coh_up)
        ax.semilogx(f_coh[ok], coh_up[ok], lw=1.5, label="coh_up")
        ax.axvspan(fmin, fmax, alpha=0.15, color="grey")
        if f_star is not None:
            ax.axvline(f_star, linestyle="--", lw=1, label=f"f*={f_star:.3f} Hz")
        ax.set_ylim(0, 1)
        ax.set_ylabel("coh_up")
        ax.legend(loc="best")

        # 3) Coherence u–w (if present)
        if coh_uw is not None:
            ax = axes[2]
            ok = np.isfinite(f_coh) & np.isfinite(coh_uw)
            ax.semilogx(f_coh[ok], coh_uw[ok], lw=1.5, label="coh_uw")
            ax.axvspan(fmin, fmax, alpha=0.15, color="grey")
            if f_star is not None:
                ax.axvline(f_star, linestyle="--", lw=1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel("coh_uw")
            ax.legend(loc="best")
        else:
            axes[1].set_xlabel("f [Hz]")

        plt.tight_layout()
        plt.show()

def plot_random_kaimal(
    pairs_by_station: Dict[str, List[Tuple[pd.Timestamp, int]]],
    base_path: str,
    *,
    k: int = 6,
    seed: int = 11,
    r_fm: float = 3.8,        # fm = r_fm * f0  (3.8 consigliato; 4 = arrotondato)
    c_const: float = 0.164,   # Kaimal constant for w-spectrum scaling (fm = c_const * U/Lw)
) -> None:
    """
    Randomly pick K (station, time, height) examples from Step-3 pairs and,
    for EACH pick, plot a 2-panel Kaimal-style figure:
      top:    normalized wind spectrum  f·Sw / σ_w^2  vs  x = f/f0
              (with Kaimal theoretical curve and x^{-2/3} slope guide)
      bottom: normalized pressure spec  f·Sp / σ_p^2  vs  x = f/f0
              (with microbarom band in x and fm = r_fm f0 line)

    f0 and fm are computed from intlenW and meanU:
        fm = c_const * U / Lw
        f0 = fm / r_fm
    """
    # Build file lists per station (same structure as your Step-3 plotter)
    files_by_station = {
        st: sorted(glob(os.path.join(base_path, st, "*.pkl")))
        for st in pairs_by_station.keys()
    }
    date_map_by_station = {st: _build_date_map(fps) for st, fps in files_by_station.items()}

    picks = _pick_random_pairs_across_stations(pairs_by_station, k=k, seed=seed)
    if not picks:
        print("No Step-3 pairs available to plot.")
        return

    for station, t, h in picks:
        fp = _find_file_for_time(station, t, files_by_station, date_map_by_station)
        if fp is None:
            print(f"[{station}] No file found for {t} h={h}")
            continue

        try:
            with open(fp, "rb") as f:
                d = pickle.load(f)
            spec: xr.Dataset  = d["spectra"]
            stats: xr.Dataset = d["stats"]
        except Exception as e:
            print(f"[{station}] failed to load {Path(fp).name}: {e}")
            continue

        t64 = np.datetime64(pd.to_datetime(t))
        h = int(h)

        # --- extract spectra
        try:
            f  = np.asarray(spec["freq"].values, dtype=float)
            sw = spec["sw"].sel(time=t64, heights=h, method="nearest").values
            sp = spec["sp"].sel(time=t64, heights=h, method="nearest").values
        except Exception as e:
            print(f"[{station}] spectra selection error at {t} h={h}: {e}")
            continue

        # --- extract stats
        try:
            U   = float(stats["meanU"].sel(time=t64, heights=h, method="nearest").values)
            ww  = float(stats["ww"].sel(time=t64, heights=h, method="nearest").values)
            pp  = float(stats["pp"].sel(time=t64, heights=h, method="nearest").values)
            Lw  = float(stats["intlenW"].sel(time=t64, heights=h, method="nearest").values)
            N2  = stats["N2"].sel(time=t64, heights=h, method="nearest").values if "N2" in stats else None
        except Exception as e:
            print(f"[{station}] stats selection error at {t} h={h}: {e}")
            continue

        if not (np.isfinite(U) and U > 0 and np.isfinite(Lw) and Lw > 0 and
                np.isfinite(ww) and ww > 0 and np.isfinite(pp) and pp > 0):
            print(f"[{station}] invalid U/Lw/variances at {t} h={h}; skipping.")
            continue

        # --- Kaimal frequencies
        fm = c_const * U / Lw
        f0 = fm / r_fm

        # Optional Brunt–Väisälä
        N = None
        if N2 is not None:
            try:
                N2 = float(N2)
                if np.isfinite(N2) and N2 > 0:
                    N = np.sqrt(N2)
            except Exception:
                N = None

        # --- dimensionless x and normalized spectra (no skipping points)
        mask = np.isfinite(f) & np.isfinite(sw) & np.isfinite(sp)
        f   = f[mask]; sw = sw[mask]; sp = sp[mask]
        x      = f / f0
        norm_SW = f * sw / ww
        norm_SP = f * sp / pp

        # microbarom band in x
        x_mb_min, x_mb_max = 0.12 / f0, 0.40 / f0

        # Kaimal theoretical curve and slope guide
        x_full = np.logspace(-2, 4, 500)
        theory_full = 0.164 * x_full / (1.0 + 0.164 * x_full ** (5.0 / 3.0))
        guide_y = x_full ** (-2.0/3.0)

        # --- figure (2 panels, shared x)
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        plt.subplots_adjust(hspace=0.05)

        # Top: wind spectrum
        ax_top.loglog(x, norm_SW, label=r'$f\,E_w(f)/\sigma_w^2$')
        ax_top.loglog(x_full, theory_full, 'k-', lw=2, label='Kaimal fit')
        ax_top.loglog(x_full, guide_y, 'r-', lw=0.8, label=r'$x^{-2/3}$ (guide)')
        ax_top.axvspan(x_mb_min, x_mb_max, color='red', alpha=0.10)
        if N is not None:
            ax_top.axvline(N / f0, linestyle='dotted', color='black', linewidth=2, label='N')
        ax_top.axvline(r_fm, color='k', ls='--', lw=1, label=rf'$f_m={r_fm}\,f_0$')
        ax_top.set_ylabel(r'$f\,E_w/\sigma_w^2$')
        ax_top.grid(True, which='both', ls='--', alpha=0.4)
        ax_top.legend(loc='best')
        ax_top.text(0.02, 0.05, 'Wind Spectra', transform=ax_top.transAxes,
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.85))

        # Bottom: pressure spectrum
        ax_bot.loglog(x, norm_SP, color='purple', label=r'$f\,E_p(f)/\sigma_p^2$')
        ax_bot.axvspan(x_mb_min, x_mb_max, color='red', alpha=0.10)
        if N is not None:
            ax_bot.axvline(N / f0, linestyle='dotted', color='red', linewidth=2, label='N')
        ax_bot.axvline(r_fm, color='k', ls='--', lw=1, label=rf'$f_m={r_fm}\,f_0$')
        ax_bot.set_xlabel(r'$f / f_0$')
        ax_bot.set_ylabel(r'$f\,E_p/\sigma_p^2$')
        ax_bot.grid(True, which='both', ls='--', alpha=0.4)
        ax_bot.legend(loc='best')
        ax_bot.text(0.02, 0.05, 'Pressure Spectrum', transform=ax_bot.transAxes,
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.85))

        # limits (stesse del tuo esempio)
        ax_top.set_xlim(1e-2, 1e4)
        ax_bot.set_xlim(1e-2, 1e4)
        ax_top.set_ylim(1e-4, 1e0)
        ax_bot.set_ylim(1e-5, 1e-1)

        day_str = pd.to_datetime(t).strftime("%Y-%m-%d")
        fig.suptitle(f"{station} | {day_str}  t={pd.to_datetime(t).strftime('%H:%M')}  h={h}  |  {Path(fp).name}", y=0.99)
        plt.tight_layout()
        plt.show()
