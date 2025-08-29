#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 08:49:01 2025

@author: mauro_ghirardelli
"""

import pickle, random
from pathlib import Path
import matplotlib.pyplot as plt
#%%

# cartella principale che contiene st1/, st2/, …
base_dir = Path("/Users/mauro_ghirardelli/Documents/TEAMx/20250416_analisi/st6")

fp1 = f"{base_dir}/2025-02-01_10min.pkl"
fp2 = f"{base_dir}/2025-02-02_10min.pkl"
fp3 = f"{base_dir}/2025-02-03_10min.pkl"
fp4 = f"{base_dir}/2025-02-04_10min.pkl"
fp5 = f"{base_dir}/2025-02-05_10min.pkl"
fp6 = f"{base_dir}/2025-02-06_10min.pkl"
fp7 = f"{base_dir}/2025-02-07_10min.pkl"



    
with open(fp7, "rb") as f:
    ds7 = pickle.load(f)
    

ds = ds7

# ── scegli il punto (ti, hi) che vuoi plottare ──────────────────────────
ti = 70       # indice tempo
hi = 0          # indice altezza




#%%

import matplotlib.pyplot as plt


# Lista delle variabili che vuoi plottare
vars_to_plot = ["su", "sv", "sw", "sp"]
labels = [r"$u$", r"$v$", r"$w$", r"$p$"]

uu_var = ds['stats']['uu'].isel(time=ti, heights=hi).values
vv_var = ds['stats']['vv'].isel(time=ti, heights=hi).values
ww_var = ds['stats']['ww'].isel(time=ti, heights=hi).values
pp_var = ds['stats']['pp'].isel(time=ti, heights=hi).values
var_dict = {"su": uu_var, "sv": vv_var, "sw": ww_var, "sp": pp_var}

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()

for ax, var, lab in zip(axes, vars_to_plot, labels):
    # RAW
    freq_raw = ds['raw_spectra']['freq'].values
    S_raw = ds['raw_spectra'][var].isel(time=ti, heights=hi).values
    norm = var_dict[var]
    y_raw = freq_raw * S_raw / norm
    
    # BINNED
    freq_bin = ds['binned_spectra']['freq'].values
    S_bin = ds['binned_spectra'][var].isel(time=ti, heights=hi).values
    y_bin = freq_bin * S_bin / norm
    
    # PROCESSED
    freq = ds['spectra']['freq'].values
    S = ds['spectra'][var].isel(time=ti, heights=hi).values
    y = freq * S / norm
    
    
    
    # Plot
    ax.loglog(freq_raw, y_raw, alpha=0.3, label="Raw")
    ax.loglog(freq_bin, y_bin,  alpha=0.3,  marker="o", ms=3, label="Binned")
    ax.loglog(freq, y, lw=2, label="Processed")
    
    
    ax.set_title(f"Spectrum {lab}")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$f \cdot S / \sigma^2$")
    ax.legend()

fig.suptitle(f"Spettri normalizzati – t index={ti}, h index={hi}", fontsize=14)
plt.tight_layout()
plt.show()
#%%
import matplotlib.pyplot as plt

# lista covarianze da plottare
covars = ["cuv", "cuw", "cvw"]   # adatta ai nomi reali nel tuo ds
labels = [r"$u'v'$", r"$u'w'$", r"$v'w'$"]

fig, axes = plt.subplots(3, 1, figsize=(4, 8), sharex=True, sharey=True)

for ax, var, lab in zip(axes, covars, labels):
    # RAW
    f_raw = ds['raw_spectra']['freq'].values
    C_raw = ds['raw_spectra'][var].isel(time=ti, heights=hi).values
    
    # PROCESSED
    f_pro = ds['spectra']['freq'].values
    C_pro = ds['spectra'][var].isel(time=ti, heights=hi).values

    # BINNED
    f_bin = ds['binned_spectra']['freq'].values
    C_bin = ds['binned_spectra'][var].isel(time=ti, heights=hi).values

    # plot
    ax.semilogx(f_raw, C_raw, alpha=0.4, label="Raw")
    ax.semilogx(f_bin, C_bin, alpha=0.6, marker="o", ms=3, label="Binned")
    ax.semilogx(f_pro, C_pro, lw=2, label="Processed")

    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"Co-spectrum {lab}")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$C(f)$")

axes[0].legend()
fig.suptitle(f"Co-spettri – t index={ti}, h index={hi}", fontsize=14)
plt.tight_layout()
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt

# ── parametri ───────────────────────────────────────────────────────────
NORMALIZE = False          # metti a False se vuoi le ogive “raw”
SHADE_BAND = (0.1, 0.6)   # banda in Hz da evidenziare

def norm_ogive(y):
    y = np.asarray(y)
    if NORMALIZE and np.isfinite(y[0]) and y[0] != 0:
        return y / y[0]
    return y

# ── estrai frequenze ───────────────────────────────────────────────────
f_raw  = ds['ogive_raw']['freq_cutoff'].values
f_bin  = ds['ogive_binned']['freq_cutoff'].values
f_smo  = ds['ogive_smoothed']['freq_cutoff'].values

# ── helper per estrarre una componente (uu, vv, ww) dai tre gruppi ─────
def get_three( field ):  # field in {'uu','vv','ww'}
    y_raw = ds['ogive_raw'][field].isel(time=ti, heights=hi).values
    y_bin = ds['ogive_binned'][field].isel(time=ti, heights=hi).values
    y_smo = ds['ogive_smoothed'][field].isel(time=ti, heights=hi).values
    return norm_ogive(y_raw), norm_ogive(y_bin), norm_ogive(y_smo)

components = ['uu','vv','ww']
titles = [r"Ogiva $u'u'$", r"Ogiva $v'v'$", r"Ogiva $w'w'$"]

# ── PLOT: 3 pannelli (uu, vv, ww) ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

for ax, comp, title in zip(axes, components, titles):
    y_raw, y_bin, y_smo = get_three(comp)

    ax.loglog(f_raw, y_raw,     lw=2, label="Raw")
    ax.loglog(f_bin, y_bin,     lw=2, marker='o', ms=3, label="Binned")
    ax.loglog(f_smo, y_smo,     lw=2, ls='--', label="Smoothed")

    ax.axvspan(SHADE_BAND[0], SHADE_BAND[1], color='red', alpha=0.25, label="0.1–0.6 Hz")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlabel("Frequenza [Hz]")

axes[0].set_ylabel("Ogiva normalizzata" if NORMALIZE else "Ogiva")
# mostra la legenda solo nel primo pannello (per evitare ripetizioni)
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, loc="best")

fig.suptitle(f"Reverse Ogive – t={ti}, h={hi}", fontsize=13)
plt.tight_layout()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

# ── parametri ───────────────────────────────────────────────────────────
NORMALIZE = False          # come nel tuo script
USE_ABS   = True           # usa il modulo per le covarianze (tipico)
SHADE_BAND = (0.1, 0.6)    # banda in Hz da evidenziare

def maybe_abs(y):
    return np.abs(y) if USE_ABS else y

def norm_ogive(y):
    y = np.asarray(y)
    if NORMALIZE and np.isfinite(y[0]) and y[0] != 0:
        return y / y[0]
    return y

# ── estrai frequenze ───────────────────────────────────────────────────
f_raw  = ds['ogive_raw']['freq_cutoff'].values
f_bin  = ds['ogive_binned']['freq_cutoff'].values
f_smo  = ds['ogive_smoothed']['freq_cutoff'].values

# ── helper per estrarre una componente (uv, uw, vw) dai tre gruppi ─────
def get_three_cov(field):  # field in {'uv','uw','vw'}
    y_raw = maybe_abs(ds['ogive_raw'][field].isel(time=ti, heights=hi).values)
    y_bin = maybe_abs(ds['ogive_binned'][field].isel(time=ti, heights=hi).values)
    y_smo = maybe_abs(ds['ogive_smoothed'][field].isel(time=ti, heights=hi).values)
    return norm_ogive(y_raw), norm_ogive(y_bin), norm_ogive(y_smo)

components = ['uv','uw','vw']
titles = [r"Ogiva $u'v'$", r"Ogiva $u'w'$", r"Ogiva $v'w'$"]

# ── PLOT: 3 pannelli (uv, uw, vw) ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

for ax, comp, title in zip(axes, components, titles):
    y_raw, y_bin, y_smo = get_three_cov(comp)

    ax.loglog(f_raw, y_raw, lw=2, label="Raw")
    ax.loglog(f_bin, y_bin, lw=2, marker='o', ms=3, label="Binned")
    ax.loglog(f_smo, y_smo, lw=2, ls='--', label="Smoothed")

    ax.axvspan(SHADE_BAND[0], SHADE_BAND[1], color='red', alpha=0.25, label="0.1–0.6 Hz")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlabel("Frequenza [Hz]")

ylabel = ("Ogiva normalizzata (modulo)" if NORMALIZE else "Ogiva (modulo)") if USE_ABS \
         else ("Ogiva normalizzata" if NORMALIZE else "Ogiva")
axes[0].set_ylabel(ylabel)

# legenda solo nel primo pannello
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, loc="best")

fig.suptitle(f"Reverse Ogive – Covarianze (t={ti}, h={hi})", fontsize=13)
plt.tight_layout()
plt.show()


#%%
import sys
sys.path.append('/Users/mauro_ghirardelli/Documents/windpy3/src/')
from spectral_analysis import spectra_processed


ds_smoothed = spectra_processed(ds7['raw_spectra'],
                      window_pts=21,
                      polyorder=3,
                      num_log_pts=1000,
                      n_bins=3000,
                      apply_binning=False)

#%%

print(ds_smoothed)
# ── scegli il punto (ti, hi) che vuoi plottare ──────────────────────────
ti = 70       # indice tempo
hi = 0          # indice altezza




#%%

