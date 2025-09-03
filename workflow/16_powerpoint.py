#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 11:08:26 2025

@author: mauro_ghirardelli
"""

#import standar libaries
import sys
import os
import numpy as np
from numpy import ma

#import third part libraries
import json
from glob import glob
import pickle
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import eig

#import local libraries
sys.path.append('/Users/mauro_ghirardelli/Documents/windpy4/src')
from filtering import find_th_pairs, find_th_pairs2, plot_th_pairs
from plotter import plot_barycentric_triangle

#%%


# === Config path ===
config_path = '/Users/mauro_ghirardelli/Documents/windpy4/conf/config_10min.txt'
with open(config_path, 'r') as file:
    config = json.load(file)

# Dizionari per accumulare i risultati
all_details = {}
all_pairs = {}

# === Itera sulle stazioni st1...st6 ===
for i in range(1, 7):
    station = f"st{i}"
    folder = f"{config['path']}{station}/"

    # Trova tutti i file pickle per questa stazione
    file_paths = sorted(glob(os.path.join(folder, "*.pkl")))
    print(f"\n--- Processing {station} ---")
    print(f"Found {len(file_paths)} pickle files.")

    # Trova le coppie th
    pairs, details = find_th_pairs(
        file_paths,
        heights_to_check=(1, 2),
        qc_threshold=0.05,
        require_fm_below_micro_min=False
    )

    print(f"Total Pairs for {station}: {len(pairs)}")

    # Salva in RAM
    all_pairs[station] = pairs
    all_details[station] = details

    # Plot se serve
    if len(pairs) > 0:
        plot_th_pairs(pairs, station)
    else:
        print(f"No pairs found for {station}.")

# Ora puoi usare all_details come mappa QC per tutto
print(all_details.keys())  # ['st1', 'st2', ..., 'st6']
#%%
print(all_details['st6'])

#%%

import json
import xarray as xr
station = 'st6'

# Load configuration file
config_path = '/Users/mauro_ghirardelli/Documents/windpy3/conf/config_10min.txt'
with open(config_path, 'r') as file:
    config = json.load(file)
    

file_path = f"{config['path']}{station}/2025-02-01_10min.pkl"


with open(file_path, 'rb') as f:
    ds = pickle.load(f)

print(ds)
#%%
# Subroutines
# -------------

import numpy as np
import matplotlib.pyplot as plt

def MB_area(
    spectrum,
    f_min=0.1,
    f_max=0.6,
    fit_deg=3,
    fit_range=(0.01, 5),
    plot=True,
    time_val=None,
    height_val=None,
    min_power=0.0  # usa un minimo >0 se vuoi evitare log di valori troppo piccoli
):
    """
    Estimate microbarom energy by fitting the background in log-log space
    outside the microbarom band and integrating inside the band.

    Parameters
    ----------
    spectrum : xr.DataArray or (freq, power) tuple
        If DataArray, must have coord 'freq'. If tuple, pass (freq_array, psd_array).
        Use an *auto*-spectrum (e.g., 'sp'), not a co-spectrum.
    f_min, f_max : float
        Microbarom band [Hz].
    fit_deg : int
        Polynomial degree for the log-log fit.
    fit_range : (float, float)
        Frequency range [Hz] where the fit is evaluated (and used outside band for fitting).
    plot : bool
        If True, show a diagnostic plot.
    time_val, height_val : optional
        Used only for title, can be strings/numbers.
    min_power : float
        Floor applied to power before log (e.g., 1e-20). Set 0.0 to disable.

    Returns
    -------
    area_fit : float
        Area under the fitted curve within the band.
    area_peak : float
        Area of the positive excess (observed - fit) within the band.
    area_peak_abs : float
        Area of the absolute deviation within the band.
    """

    # ---- Ingest ----
    if isinstance(spectrum, tuple) and len(spectrum) == 2:
        freq, sp = spectrum
    else:
        # xarray.DataArray atteso
        if not hasattr(spectrum, "freq"):
            raise ValueError("If 'spectrum' is a DataArray it must have a 'freq' coordinate.")
        freq = spectrum.freq.values
        sp = spectrum.values

    # Cast sicuro a float reale
    freq = np.asarray(freq, dtype=float)
    sp = np.asarray(np.real(sp), dtype=float)

    # Auto-spettri: devono essere >= 0; tolleranza numerica
    if min_power is not None and min_power > 0.0:
        sp = np.maximum(sp, float(min_power))

    # Filtro validi
    valid = (np.isfinite(freq) & np.isfinite(sp) & (freq > 0))
    freq_valid = freq[valid]
    sp_valid = sp[valid]

    if freq_valid.size < fit_deg + 1:
        return np.nan, np.nan, np.nan

    # ---- Maschere di fit ----
    fit_mask = (
        (freq_valid >= fit_range[0]) &
        (freq_valid <= fit_range[1]) &
        ((freq_valid < f_min) | (freq_valid > f_max))
    )
    f_fit = freq_valid[fit_mask]
    sp_fit = sp_valid[fit_mask]

    if f_fit.size < fit_deg + 1:
        return np.nan, np.nan, np.nan

    # ---- Fit in log-log ----
    logf = np.log10(f_fit)
    logsp = np.log10(sp_fit)
    coeffs = np.polyfit(logf, logsp, deg=fit_deg)

    # Valutazione del fit nel range desiderato
    fit_eval_mask = (freq_valid >= fit_range[0]) & (freq_valid <= fit_range[1])
    freq_eval = freq_valid[fit_eval_mask]
    sp_eval = sp_valid[fit_eval_mask]
    sp_fit_all = 10.0 ** np.polyval(coeffs, np.log10(freq_eval))

    # ---- Banda microbarom ----
    band_mask = (freq_eval >= f_min) & (freq_eval <= f_max)
    f_band = freq_eval[band_mask]
    sp_band = sp_eval[band_mask]
    fit_band = sp_fit_all[band_mask]

    if f_band.size == 0:
        return np.nan, np.nan, np.nan

    # ---- Aree ----
    area_fit = np.trapz(fit_band, x=f_band)
    pos_excess = np.maximum(sp_band - fit_band, 0.0)
    area_peak = np.trapz(pos_excess, x=f_band)
    area_peak_abs = np.trapz(np.abs(sp_band - fit_band), x=f_band)

    # ---- Plot ----
    if plot:
        title_bits = []
        if time_val is not None:  title_bits.append(f"Time: {time_val}")
        if height_val is not None: title_bits.append(f"Height: {height_val} m")
        title = " | ".join(title_bits) if title_bits else "Microbarom band analysis"

        plt.figure(figsize=(8, 4))
        plt.loglog(freq_valid, sp_valid, label='Spectrum', alpha=0.75)
        # fit su tutta la griglia valida (per visual)
        full_fit = 10.0 ** np.polyval(coeffs, np.log10(freq_valid))
        plt.loglog(freq_valid, full_fit, '--', label='Log-Log Fit', alpha=0.9)
        plt.axvspan(f_min, f_max, color='gray', alpha=0.3, label='Microbarom band')
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("S_pp")
        #plt.title(title)
        plt.grid(True, which='both', ls='--', lw=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return area_fit, area_peak, area_peak_abs


# 1) scegli un time (es. indice 0) e un'altezza (se presente)
DA = ds["spectra"]["sp"].isel(time=35)
if "heights" in DA.dims:
    DA = DA.isel(heights=0)

# 2) chiama la funzione
area_fit, area_peak, area_peak_abs = MB_area(
    DA,
    f_min=0.1, f_max=0.6,
    fit_deg=3, fit_range=(0.01, 5),
    plot=True,
    time_val=str(DA.time.values) if "time" in DA.coords else None,
    height_val=float(DA.heights.values) if "heights" in DA.coords else None,
    min_power=1e-20   # evita log di 0
)

#%%
import numpy as np
import matplotlib.pyplot as plt

def MB_area(
    spectrum,
    f_min=0.1,
    f_max=0.6,
    fit_deg=3,
    fit_range=(0.01, 5),
    plot=True,
    time_val=None,
    height_val=None,
    min_power=0.0
):
    """
    Compute microbarom band energy and visualize:
    - Green vertical lines and gray band to highlight microbarom range
    - Red area between spectrum and fit (excess)
    - Light gray fill under the fit curve down to the bottom of the plot

    Parameters
    ----------
    spectrum : xarray.DataArray or (freq, power) tuple
        Spectrum (1D over 'freq') or (freq, values) tuple.
    f_min, f_max : float
        Microbarom band [Hz].
    fit_deg : int
        Polynomial degree for log-log fit.
    fit_range : (float, float)
        Frequency range [Hz] for fit evaluation.
    plot : bool
        If True, plot.
    time_val, height_val : optional
        Metadata for title (str/num).
    min_power : float
        Floor for power values to avoid log10(0).
    """
    # ---- Input handling ----
    if isinstance(spectrum, tuple) and len(spectrum) == 2:
        freq, sp = spectrum
        freq = np.asarray(freq)
        sp = np.asarray(sp)
    else:
        DA = spectrum
        if hasattr(DA, "squeeze"):
            for d in list(DA.dims):
                if d != "freq":
                    DA = DA.squeeze(d, drop=True)
        if "freq" not in getattr(DA, "dims", []):
            raise ValueError("Spectrum must be 1D over 'freq' or provide (freq, S).")
        freq = DA.freq.values
        sp = DA.values

    freq = np.asarray(freq, dtype=float).ravel()
    sp = np.asarray(np.real(sp), dtype=float).ravel()
    if freq.size != sp.size:
        raise ValueError(f"Length mismatch: freq={freq.size}, S={sp.size}")

    if min_power > 0.0:
        sp = np.maximum(sp, float(min_power))

    valid = np.isfinite(freq) & np.isfinite(sp) & (freq > 0)
    f = freq[valid]
    S = sp[valid]
    if f.size < fit_deg + 1:
        return np.nan, np.nan, np.nan

    # ---- Fit selection ----
    mask_fit = (f >= fit_range[0]) & (f <= fit_range[1]) & ((f < f_min) | (f > f_max))
    f_fit = f[mask_fit]; S_fit = S[mask_fit]
    if f_fit.size < fit_deg + 1:
        return np.nan, np.nan, np.nan

    coeffs = np.polyfit(np.log10(f_fit), np.log10(S_fit), deg=fit_deg)
    S_model = 10 ** np.polyval(coeffs, np.log10(f))

    # ---- Band ----
    mask_band = (f >= f_min) & (f <= f_max)
    if not np.any(mask_band):
        return np.nan, np.nan, np.nan

    f_band = f[mask_band]
    S_band = S[mask_band]
    fit_band = S_model[mask_band]

    # ---- Areas ----
    area_fit = np.trapz(fit_band, x=f_band)
    area_excess = np.trapz(np.maximum(S_band - fit_band, 0.0), x=f_band)
    area_abs_diff = np.trapz(np.abs(S_band - fit_band), x=f_band)

    # ---- Plot ----
    if plot:
        title_bits = []
        if time_val is not None: title_bits.append(f"Time: {time_val}")
        if height_val is not None: title_bits.append(f"H: {height_val} m")
        title = " | ".join(title_bits) if title_bits else "Microbarom Band Analysis"

        plt.figure(figsize=(9, 5))
        plt.loglog(f, S, label="Spectrum", alpha=0.75)
        plt.loglog(f, S_model, "--", label="Log-Log Fit", alpha=0.95)

        # Band highlight: vertical lines and gray zone
        plt.axvline(f_min, color="green", linestyle="--", lw=1.5, label="Microbarom bounds")
        plt.axvline(f_max, color="green", linestyle="--", lw=1.5)
        #plt.axvspan(f_min, f_max, color="gray", alpha=0.15)

        # Fill under fit down to the plot bottom
        plt.fill_between(f_band, 1e-20, fit_band, color="lightgray", alpha=0.5, label="Fit baseline")

        # Red area for excess over fit
        plt.fill_between(f_band, fit_band, S_band, where=S_band > fit_band,
                         color="red", alpha=0.45, label="Excess over fit")

        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD pressure [Pa²/Hz]")
        plt.grid(True, which="both", ls="--", lw=0.3)
        plt.legend()
        plt.ylim(10e-010,10e-2)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    return area_fit, area_excess, area_abs_diff



f = DA.freq.values
S = DA.values
MB_area((f, S), f_min=0.1, f_max=0.6, fit_deg=3, fit_range=(0.01, 5), plot=True)
