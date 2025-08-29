
#import standar libaries
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

#import third part libraries
import pickle
import pandas as pd
from typing import Iterable, Tuple, Optional, List, Any, Dict







def find_th_pairs2(
    file_paths: Iterable[Path],
    heights_to_check: Tuple[int, ...],
    *,
    qc_threshold: Optional[float] = None,
    qc_op: str = "<=",
    I_threshold: Optional[float] = None,
    I_op: str = "<=",
    require_meanU_not_all_nan: bool = True,
    time_range: Optional[Tuple[Any, Any]] = None,   # (t_min, t_max) comparable with stats["time"].values
    station_regex: Optional[str] = r"(st\d+)",      # if None, station is not extracted
    return_details: bool = True,
    zL_abs_max: Optional[float] = None,   # NEW: keep only |z/L| ≤ zL_abs_max (if provided)
):
    """
    Return (time, height) pairs that satisfy:

      • Height is present (in the file) and, if requested, the meanU column is not all-NaN
      • QCnan vs qc_threshold (operator configurable), only if qc_threshold is provided
      • Turbulence intensity I vs I_threshold (operator configurable), only if I_threshold is provided
      • Time falls within `time_range`, only if `time_range` is provided
      • Neutrality filter via z/L: keep only points with |z/L| ≤ zL_abs_max, if provided.


    Notes
    -----
    Only the conditions for which a key-argument is provided are enforced.
    For example, if `qc_threshold` is None, the QC filter is skipped; if `I_threshold`
    is None, the I filter is skipped. If *no* explicit conditions are provided, the
    function returns all (time, height) pairs available at the requested heights
    (optionally filtering out heights whose `meanU` is all-NaN if
    `require_meanU_not_all_nan=True`).

    Parameters
    ----------
    file_paths : list[Path]
        .pkl files to scan.
    heights_to_check : tuple[int]
        Heights (m) to evaluate, e.g. (1,) or (1, 2).
    qc_threshold : float or None, optional
        Threshold for QCnan. If None, QC is not used as a filter.
        Default None.
    qc_op : {"<", "<=", ">", ">=", "==", "!="}, optional
        Comparison operator applied as QCnan {op} qc_threshold.
        Default "<=".
    I_threshold : float or None, optional
        Threshold for turbulence intensity I. If None, I is not used as a filter.
        Default None.
    I_op : {"<", "<=", ">", ">=", "==", "!="}, optional
        Comparison operator applied as I {op} I_threshold.
        Default "<=".
    require_meanU_not_all_nan : bool, optional
        If True, a height is considered only if its meanU column contains at least one
        non-NaN value across the file. If False, this pre-check is skipped.
        Default True.
    time_range : tuple or None, optional
        (t_min, t_max). If provided, only times t with t_min ≤ t ≤ t_max are considered.
        Default None.
    station_regex : str or None, optional
        Regex used to extract a station label from the file path (e.g., "st1").
        If None, station is not extracted. Default r"(st\\d+)".
    return_details : bool, optional
        If True, also return a pandas DataFrame with raw values and boolean outcomes
        for each applied condition. Default True.

    Returns
    -------
    list[tuple]
        Sorted list of unique (time, height) pairs that pass all the **provided**
        conditions.
    pandas.DataFrame (optional)
        When `return_details=True`, a DataFrame with columns:
        ["time", "height", "station", "file", "meanU", "QCnan", "I", "passed"].
        `passed` indicates whether the row satisfied all provided conditions.

    Examples
    --------
    # Apply both QC and I filters
    pairs, details = find_th_pairs(files, (1, 2), qc_threshold=0.10, I_threshold=0.50)

    # Apply only QC filter (skip I)
    pairs, details = find_th_pairs(files, (1, 2), qc_threshold=0.10)

    # No QC/I filters, keep only heights whose meanU is not all-NaN
    pairs, details = find_th_pairs(files, (1, 2))
    """
    OPS = {
        "<":  lambda a, b: a <  b,
        "<=": lambda a, b: a <= b,
        ">":  lambda a, b: a >  b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }
    if qc_threshold is not None and qc_op not in OPS:
        raise ValueError(f"Invalid qc_op: {qc_op}")
    if I_threshold is not None and I_op not in OPS:
        raise ValueError(f"Invalid I_op: {I_op}")

    valid_pairs = set()
    rows: List[Dict[str, Any]] = []

    for path in file_paths:
        with open(path, "rb") as f:
            ds = pickle.load(f)
        stats = ds["stats"]

        times    = stats["time"].values                # (t,)
        heights  = stats["heights"].values.astype(int) # (h,)
        mean_u   = stats["meanU"].values               # (t,h)
        uu, vv, ww, wT = (stats[k].values for k in ("uu", "vv", "ww", "wT"))
        qc_nan   = stats["QCnan"].values               # (t,h)

        # Turbulence intensity
        I = np.sqrt(uu**2 + vv**2 + ww**2) / mean_u
        
        # --- Monin–Obukhov length L (compute only if zL filter is requested) ---
        L = None
        if zL_abs_max is not None:
            if all(k in stats for k in ("ustar", "wT", "meanT")):
                ustar = stats["ustar"].values   # shape (t,) or (t,h)
                wT    = stats["wT"].values      # "
                meanT = stats["meanT"].values   # "

                # se 2D, riduci a serie temporali (fallback innocuo)
                if ustar.ndim == 2: ustar = np.nanmean(ustar, axis=1)
                if wT.ndim    == 2: wT    = np.nanmean(wT,    axis=1)
                if meanT.ndim == 2: meanT = np.nanmean(meanT, axis=1)

                theta = meanT + 273.15 if np.nanmean(meanT) < 200 else meanT  # K
                kappa, g = 0.4, 9.81
                eps = 1e-6
                denom = kappa * g * wT
                denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps, denom)
                L = -(ustar**3 * theta) / denom   # shape (t,)
            # else: lascia L=None → filtro zL verrà ignorato


        # Optional station extraction
        station = None
        if station_regex:
            m = re.search(station_regex, str(path))
            if m:
                station = m.group(1)

        # Map height -> index
        idx_by_height = {int(h): i for i, h in enumerate(heights)}

        # Pre-select usable heights
        usable = []
        for h in heights_to_check:
            hi = idx_by_height.get(int(h))
            if hi is None:
                continue
            if require_meanU_not_all_nan and np.all(np.isnan(mean_u[:, hi])):
                continue
            usable.append((int(h), hi))

        # Iterate times/heights
        for ti, t in enumerate(times):
            if time_range is not None:
                tmin, tmax = time_range
                if not (tmin <= t <= tmax):
                    continue

            for h, hi in usable:
                mu   = mean_u[ti, hi]
                qc   = qc_nan[ti, hi]
                Ival = I[ti, hi]

                conds = []

                # QC filter (only if threshold provided and QC is finite)
                if qc_threshold is not None:
                    if np.isfinite(qc):
                        conds.append(OPS[qc_op](qc, qc_threshold))
                    else:
                        conds.append(False)

                # I filter (only if threshold provided and I is finite)
                if I_threshold is not None:
                    if np.isfinite(Ival):
                        conds.append(OPS[I_op](Ival, I_threshold))
                    else:
                        conds.append(False)

                # z/L filter (only if threshold provided and L is finite at time ti)
                if zL_abs_max is not None and (L is not None) and np.isfinite(L[ti]):
                    zL_val = h / L[ti]
                    conds.append(np.isfinite(zL_val) and (abs(zL_val) <= zL_abs_max))
                else:
                    zL_val = np.nan  # opzionale, utile per salvare nei details

                # If no explicit conditions were provided, accept by default
                passed = all(conds) if conds else True

                if passed:
                    valid_pairs.add((t, h))

                rows.append({
                    "time": t,
                    "height": h,
                    "station": station,
                    "file": str(path),
                    "meanU": mu,
                    "QCnan": qc,
                    "I": Ival,
                    "passed": passed,
                    "L": (np.nan if L is None else L[ti]),
                    "zL": zL_val,

                })

        del ds  # free memory

    pairs = sorted(valid_pairs)
    if return_details:
        details = pd.DataFrame(rows)
        if not details.empty:
            details.sort_values(["passed", "time", "height", "station"], ascending=[False, True, True, True], inplace=True)
        return pairs, details
    else:
        return pairs

def find_th_pairs(
    file_paths: Iterable[Path],
    heights_to_check: Tuple[int, ...],
    *,
    qc_threshold: Optional[float] = None,
    qc_op: str = "<=",
    I_threshold: Optional[float] = None,
    I_op: str = "<=",
    require_meanU_not_all_nan: bool = True,
    time_range: Optional[Tuple[Any, Any]] = None,
    station_regex: Optional[str] = r"(st\d+)",
    return_details: bool = True,
    zL_abs_max: Optional[float] = None,
    # --------- NUOVI PARAMETRI ---------
    require_fm_below_micro_min: bool = False,
    fm_normalized: float = 3.8,
):
    """
    [...] (docstring invariata)

    Filtro extra (opzionale)
    ------------------------
    Se `require_fm_below_micro_min=True`, per ciascun (time,height) richiede:

        fm_normalized < f_micro_min

    dove:
        U = stats['meanU']
        timelenW = stats['intlenW'] / U
        f0 = (0.164/4) * (1.0 / timelenW)
        f_micro_min  = 0.12 / f0
        f_micro_max  = 0.40 / f0
        f_micro_mean = 0.26 / f0

    I valori f0 e affini sono calcolati elemento per elemento (t,h).
    """
    OPS = {
        "<":  lambda a, b: a <  b,
        "<=": lambda a, b: a <= b,
        ">":  lambda a, b: a >  b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }
    if qc_threshold is not None and qc_op not in OPS:
        raise ValueError(f"Invalid qc_op: {qc_op}")
    if I_threshold is not None and I_op not in OPS:
        raise ValueError(f"Invalid I_op: {I_op}")

    valid_pairs = set()
    rows: List[Dict[str, Any]] = []

    for path in file_paths:
        with open(path, "rb") as f:
            ds = pickle.load(f)
        stats = ds["stats"]

        times    = stats["time"].values
        heights  = stats["heights"].values.astype(int)
        mean_u   = stats["meanU"].values               # (t,h)
        uu, vv, ww, wT = (stats[k].values for k in ("uu", "vv", "ww", "wT"))
        qc_nan   = stats["QCnan"].values               # (t,h)

        # Turbulence intensity
        I = np.sqrt(uu**2 + vv**2 + ww**2) / mean_u

        # --- Monin–Obukhov length (come prima) ---
        L = None
        if zL_abs_max is not None:
            if all(k in stats for k in ("ustar", "wT", "meanT")):
                ustar = stats["ustar"].values
                wT_   = stats["wT"].values
                meanT = stats["meanT"].values
                if ustar.ndim == 2: ustar = np.nanmean(ustar, axis=1)
                if wT_.ndim    == 2: wT_  = np.nanmean(wT_,    axis=1)
                if meanT.ndim  == 2: meanT= np.nanmean(meanT,  axis=1)
                theta = meanT + 273.15 if np.nanmean(meanT) < 200 else meanT
                kappa, g = 0.4, 9.81
                eps = 1e-12
                denom = kappa * g * wT_
                denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps, denom)
                L = -(ustar**3 * theta) / denom
            # else L resta None

        # --- NUOVO: f0 e soglie microbarom normalizzate (per (t,h)) ---
        if require_fm_below_micro_min:
            # intlenW può essere (t,) o (t,h) → broadcast a (t,h)
            intlenW = stats["intlenW"].values
            if intlenW.ndim == 1:
                intlenW = intlenW[:, None]  # (t,1)
            try:
                intlenW = np.broadcast_to(intlenW, mean_u.shape)
            except Exception:
                # fallback: ripeti lungo seconda dim
                intlenW = np.repeat(intlenW, mean_u.shape[1], axis=1)

            with np.errstate(divide="ignore", invalid="ignore"):
                timelenW = intlenW / mean_u                      # (t,h)
                f0 = (0.164/4.0) * (1.0 / timelenW)             # (t,h)
                f_micro_min  = 0.12 / f0                        # (t,h)
                f_micro_max  = 0.40 / f0                        # (t,h)
                f_micro_mean = 0.26 / f0                        # (t,h)
            # proteggi da divisioni/NaN/inf
            f_micro_min  = np.where(np.isfinite(f_micro_min),  f_micro_min,  np.nan)
            f_micro_max  = np.where(np.isfinite(f_micro_max),  f_micro_max,  np.nan)
            f_micro_mean = np.where(np.isfinite(f_micro_mean), f_micro_mean, np.nan)
        else:
            f0 = f_micro_min = f_micro_max = f_micro_mean = None

        # Optional station extraction
        station = None
        if station_regex:
            m = re.search(station_regex, str(path))
            if m: station = m.group(1)

        # Map height -> index
        idx_by_height = {int(h): i for i, h in enumerate(heights)}

        # Pre-select usable heights
        usable = []
        for h in heights_to_check:
            hi = idx_by_height.get(int(h))
            if hi is None:
                continue
            if require_meanU_not_all_nan and np.all(np.isnan(mean_u[:, hi])):
                continue
            usable.append((int(h), hi))

        for ti, t in enumerate(times):
            if time_range is not None:
                tmin, tmax = time_range
                if not (tmin <= t <= tmax):
                    continue

            for h, hi in usable:
                mu   = mean_u[ti, hi]
                qc   = qc_nan[ti, hi]
                Ival = I[ti, hi]

                conds = []

                # QC
                if qc_threshold is not None:
                    conds.append(np.isfinite(qc) and OPS[qc_op](qc, qc_threshold))

                # I
                if I_threshold is not None:
                    conds.append(np.isfinite(Ival) and OPS[I_op](Ival, I_threshold))

                # z/L
                if zL_abs_max is not None and (L is not None) and np.isfinite(L[ti]):
                    zL_val = h / L[ti]
                    conds.append(np.isfinite(zL_val) and (abs(zL_val) <= zL_abs_max))
                else:
                    zL_val = np.nan

                # --- NUOVO filtro INERTIAL: fm_normalized < f_micro_min ---
                if require_fm_below_micro_min:
                    fmin_here  = f_micro_min[ti, hi]
                    inertial_ok = (np.isfinite(fmin_here) and (fm_normalized < fmin_here))
                    conds.append(inertial_ok)
                else:
                    inertial_ok = np.nan

                passed = all(conds) if conds else True
                if passed:
                    valid_pairs.add((t, h))

                row = {
                    "time": t,
                    "height": h,
                    "station": station,
                    "file": str(path),
                    "meanU": mu,
                    "QCnan": qc,
                    "I": Ival,
                    "passed": passed,
                    "L": (np.nan if L is None else L[ti]),
                    "zL": zL_val,
                }
                # colonne diagnostiche nuove
                if require_fm_below_micro_min:
                    row.update({
                        "f0": f0[ti, hi],
                        "f_micro_min": fmin_here,
                        "f_micro_max": f_micro_max[ti, hi],
                        "f_micro_mean": f_micro_mean[ti, hi],
                        "fm_norm": fm_normalized,
                        "inertial_ok": inertial_ok,
                    })
                rows.append(row)

        del ds  # free memory

    pairs = sorted(valid_pairs)
    if return_details:
        details = pd.DataFrame(rows)
        if not details.empty:
            details.sort_values(
                ["passed", "time", "height", "station"],
                ascending=[False, True, True, True],
                inplace=True
            )
        return pairs, details
    else:
        return pairs



def plot_th_pairs(
    pairs,
    station_name="",
    *,
    height_offsets=None,      # dict {height: hour-shift} or None → auto
    marker_size=6,
    alpha=0.9,
    figsize=None,
    jitter_minutes=0.0,       # extra random jitter in minutes to reduce overplotting
    color_map=None,           # dict {height: color} or None → auto cycle
    sort_days="asc",          # "asc" | "desc"
    show_legend=True,
):
    """
    Visualize valid (time, height) pairs as points along the day axis.

    Parameters
    ----------
    pairs : list[tuple]
        List of (time, height) tuples. `time` must be datetime-like or convertible.
    station_name : str, optional
        Title prefix for the plot.
    height_offsets : dict or None, optional
        Horizontal offset in hours applied per height, e.g. {1: -0.04, 2: 0.04}.
        If None, offsets are generated automatically to avoid overlap.
    marker_size : int, optional
        Marker size for scatter points. Default 6.
    alpha : float, optional
        Marker opacity. Default 0.9.
    figsize : tuple or None, optional
        Matplotlib figsize. If None, computed from number of days.
    jitter_minutes : float, optional
        Add ±uniform jitter in minutes to time-of-day to reduce overplotting.
        Default 0.0 (disabled).
    color_map : dict or None, optional
        Colors per height. If None, use Matplotlib default cycle.
    sort_days : {"asc","desc"}, optional
        Sort order for day rows. Default "asc".
    show_legend : bool, optional
        Whether to show a legend mapping color to height. Default True.

    Returns
    -------
    None
        Shows a matplotlib figure.
    """
    if not pairs:
        print(f"{station_name}: no data to plot.")
        return

    # Build DataFrame
    df = pd.DataFrame(pairs, columns=["time", "height"])
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    df["hour_float"] = df["time"].dt.hour + df["time"].dt.minute / 60.0 + df["time"].dt.second / 3600.0

    # Day order
    days = np.sort(df["date"].unique())
    if sort_days == "desc":
        days = days[::-1]
    day_to_y = {d: i for i, d in enumerate(days)}
    df["y"] = df["date"].map(day_to_y)

    # Height offsets
    heights = sorted(df["height"].unique())
    if height_offsets is None:
        # spread offsets evenly around 0 (in hours)
        # e.g. for 2 heights -> [-0.04, +0.04]; for 3 -> [-0.06, 0, +0.06]
        base = 0.04  # ~2.4 minutes
        k = len(heights)
        if k == 1:
            offsets = [0.0]
        else:
            # symmetric sequence around 0
            half = (k - 1) / 2
            offsets = [base * (i - half) for i in range(k)]
        height_offsets = dict(zip(heights, offsets))
    df["x"] = df["hour_float"] + df["height"].map(height_offsets)

    # Optional jitter
    if jitter_minutes and jitter_minutes > 0:
        jitter = (np.random.rand(len(df)) - 0.5) * (2 * jitter_minutes) / 60.0
        df["x"] = df["x"] + jitter

    # Colors
    if color_map is None:
        # assign colors from default cycle
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        color_map = {}
        for i, h in enumerate(heights):
            color_map[h] = prop_cycle[i % max(1, len(prop_cycle))]
    df["color"] = df["height"].map(color_map)

    # Figure size
    if figsize is None:
        figsize = (20, max(6, int(len(days) * 0.4)))

    # Plot (vectorized per-height)
    fig, ax = plt.subplots(figsize=figsize)
    for h in heights:
        sub = df[df["height"] == h]
        ax.scatter(sub["x"], sub["y"], s=marker_size, alpha=alpha, label=f"h={h} m", c=sub["color"])

    # Axes formatting
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 1))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 1)])
    ax.set_yticks(range(len(days)))
    ax.set_yticklabels([pd.to_datetime(d).strftime("%Y-%m-%d") for d in days])
    ax.set_xlabel("Time of day")
    n_points = len(df)
    title_suffix = f" ({n_points} points; {len(days)} days; heights={heights})"
    ax.set_title(f"{station_name}: valid windows{title_suffix}")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    if show_legend and len(heights) > 1:
        ax.legend(title="Height", loc="upper right", frameon=False)

    fig.tight_layout()
    plt.show()







