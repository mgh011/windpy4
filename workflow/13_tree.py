#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 13:50:35 2025

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
sys.path.append('/Users/mauro_ghirardelli/Documents/windpy3/src')
from filtering import find_th_pairs, find_th_pairs2, plot_th_pairs
from plotter import plot_barycentric_triangle

#%%

# Load configuration file
config_path = '/Users/mauro_ghirardelli/Documents/windpy3/conf/config_10min.txt'
with open(config_path, 'r') as file:
    config = json.load(file)
    
   

station = "st2"
folder = f"{config['path']}{station}/"
file_paths_st2 = sorted(glob(os.path.join(folder, "*.pkl")))


pairs_st2, details_st2 = find_th_pairs(
    file_paths_st2,
    heights_to_check=(1, 2),
    qc_threshold=0.05,   # QCnan must be ≤ 0.1
    #I_threshold = 0.5,
    #zL_abs_max=0.05
    #require_fm_below_micro_min=True
)

print(f"Total Files: {len(file_paths_st2)}")
print(f"Total Pairs: {len(pairs_st2)}")   

plot_th_pairs(pairs_st2, "st2")
#%%

#
#
# -------------- BLOCCO I: MB PEAK
#
#
#


# --- helper robust z ---
def robust_z(x):
    med = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med))
    z = (x - med) / (mad if mad > 0 else 1e-12)
    return z, med, mad

# --- 1) mappa {date_str: filepath} (assumo 1 file per giorno) ---
def date_from_path(fp):
    # estrae 'YYYY-MM-DD' dal nome file '.../2025-02-14_10min.pkl'
    base = fp.split(os.sep)[-1]
    return base.split('_')[0]  # '2025-02-14'

date2fp = {date_from_path(fp): fp for fp in file_paths_st2}

# --- 2) raggruppa le coppie per giorno per evitare mille reopen ---
pairs_by_date = {}
for ph in pairs_st2:
    t, h = ph  # tupla: (numpy.datetime64, int)
    # ricava stringa data 'YYYY-MM-DD' dal timestamp
    date_str = pd.to_datetime(t).strftime('%Y-%m-%d')
    pairs_by_date.setdefault(date_str, []).append((t, h))

# --- 3) estrai MB solo per le coppie ---
rows = []
cache = {}  # cache dei file caricati per giorno

for date_str, pairs in pairs_by_date.items():
    fp = date2fp.get(date_str)
    if fp is None:
        # nessun file per quel giorno (capita a cavallo di mesi, ecc.)
        continue
    if date_str not in cache:
        with open(fp, 'rb') as f:
            cache[date_str] = pickle.load(f)
    d = cache[date_str]
    MB = d["stats"]["MB_peak"] / d["stats"]["MB_fit"]  # xarray.DataArray (time, heights)

    for (t, h) in pairs:
        try:
            # selezione esatta su coordinate
            mb_val = MB.sel(time=np.datetime64(t), heights=int(h)).item()
        except Exception:
            # se per arrotondamenti serve nearest:
            mb_val = MB.sel(time=np.datetime64(t), heights=int(h), method="nearest").item()
        rows.append({"time": pd.to_datetime(t), "height": int(h), "MB_value": float(mb_val)})

# DataFrame con SOLO le coppie valide
df_mb = pd.DataFrame(rows).sort_values(["time","height"]).reset_index(drop=True)

# --- 4) z-robusto e flag soglia ---
z, med, mad = robust_z(df_mb["MB_value"].values)
df_mb["MB_z"] = z

tau_z = 3.0  # default; prova 2.5 (lenient) o 3.5 (strict) se serve
df_mb["MB_flag"] = df_mb["MB_z"] >= tau_z

print("Baseline MB median:", med)
print("Baseline MB MAD:", mad)
print("Total pairs:", len(df_mb))
print("Flagged (MB):", int(df_mb["MB_flag"].sum()))

#%%

import matplotlib.pyplot as plt

# Istogramma di MB_value
plt.figure(figsize=(5,3))
plt.hist(df_mb["MB_value"], bins=50, alpha=0.6, color="skyblue", edgecolor="k")

# Linea mediana
plt.axvline(med, color="red", linestyle="--", lw=2, label=f"Median = {med:.2e}")

# Linee soglia z = ±3
low_thr  = med - 3*mad
high_thr = med + 3*mad
plt.axvline(low_thr, color="orange", linestyle=":", lw=2, label=f"-3 MAD = {low_thr:.2e}")
plt.axvline(high_thr, color="orange", linestyle=":", lw=2, label=f"+3 MAD = {high_thr:.2e}")

plt.xlabel("MB_value (MB_peak / MB_fit)")
plt.ylabel("Events")
plt.title(" MB_value distribution robust-z")
plt.xlim(-1, 10)
plt.legend()
plt.tight_layout()
plt.show()

#%%
import pandas as pd
import numpy as np

# df_mb has columns: time (datetime64), height (int), MB_value, MB_z, MB_flag

# --- ensure sorted ---
df_mb = df_mb.sort_values(['height','time']).reset_index(drop=True)

# helper: compute run lengths for consecutive 10-min steps
def add_temporal_persistence_flags(df, step='10min'):
    df = df.sort_values(['height','time']).copy()
    df['MB_flag_persist'] = False

    for h, g in df.groupby('height', group_keys=False):
        # consecutività avanti/indietro
        consec_prev = g['time'].diff().eq(pd.Timedelta(step))          # tra t-1 → t
        consec_next = g['time'].shift(-1).sub(g['time']).eq(pd.Timedelta(step))  # tra t → t+1

        # vicini MB (consecutivi) su almeno un lato
        left_ok  = g['MB_flag'].shift(1).fillna(False)  & consec_prev
        right_ok = g['MB_flag'].shift(-1).fillna(False) & consec_next

        persist_here = g['MB_flag'] & (left_ok | right_ok)
        df.loc[g.index, 'MB_flag_persist'] = persist_here

    return df


df_mb = add_temporal_persistence_flags(df_mb, step='10min')

# df_mb has: time, height, MB_flag, MB_runlen, MB_flag_persist

# count how many heights are persistent at each timestamp
counts_at_t = (
    df_mb.loc[df_mb['MB_flag_persist']]
         .groupby('time')['height']
         .nunique()
)

# robust mapping (works even if counts_at_t is empty)
df_mb['MB_heights_at_time'] = df_mb['time'].map(counts_at_t).fillna(0).astype(int)

# final flag (toggle vertical concurrence as you like)
require_vertical = True  # True => require ≥2 heights at same time
if require_vertical:
    df_mb['MB_flag_final'] = df_mb['MB_flag_persist'] & (df_mb['MB_heights_at_time'] >= 2)
else:
    df_mb['MB_flag_final'] = df_mb['MB_flag_persist']

print({
    "flag_raw": int(df_mb['MB_flag'].sum()),
    "flag_persistent": int(df_mb['MB_flag_persist'].sum()),
    "flag_final": int(df_mb['MB_flag_final'].sum()),
})

#%%
rng = np.random.default_rng(11)  # per riproducibilità

# --- helper: prendi 'YYYY-MM-DD' dal path ---
def date_from_path(fp):
    base = fp.split(os.sep)[-1]
    return base.split('_')[0]  # '2025-02-14'

date2fp = {date_from_path(fp): fp for fp in file_paths_st2}

# --- scegli k esempi tra i flaggati ---
k = 9  # quanti plottare
df_flag = df_mb.loc[df_mb['MB_flag_final']].copy()
if df_flag.empty:
    print("Nessun MB_flag_final=True da plottare.")
else:
    pick_idx = rng.choice(df_flag.index.values, size=min(k, len(df_flag)), replace=False)
    df_pick = df_flag.loc[pick_idx].sort_values(['time','height'])

    ncols = 3
    nrows = int(np.ceil(len(df_pick)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)

    for ax, (_, row) in zip(axes.ravel(), df_pick.iterrows()):
        t = row['time']
        h = int(row['height'])
        day = pd.to_datetime(t).strftime('%Y-%m-%d')
        fp = date2fp.get(day, None)
        if fp is None:
            ax.set_visible(False)
            continue

        with open(fp, 'rb') as f:
            d = pickle.load(f)

        # estrai spettro pressione e frequenze
        spec = d['spectra']
        fHz = spec['freq'].values  # (F,)
        Spp = spec['sp'].sel(time=np.datetime64(t), heights=h, method='nearest').values  # (F,)
        # sicurezza: niente <=0 per log
        mask_pos = (Spp > 0) & (fHz > 0)
        f = fHz[mask_pos]; P = Spp[mask_pos]

        # plot log-log
        ax.loglog(f, f*P, lw=1.5)
        ax.set_xlabel('f [Hz]'); ax.set_ylabel('f*Spp')
        ax.set_title(f"{day}  t={pd.to_datetime(t).strftime('%H:%M')}  h={h}\n"
                     f"MB={row['MB_value']:.2e}  z={row['MB_z']:.2f}")

        # evidenzia banda MB
        fmin, fmax = 0.1, 0.4
        ax.axvspan(fmin, fmax, alpha=0.15)
        ax.set_xlim(left=max(f.min(), 1e-3), right=min(f.max(), 10))


    # nascondi subplot vuoti
    for ax in axes.ravel()[len(df_pick):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()
#%%
    # elenco di coppie (time, height) che hanno superato lo Step-1
pairs_st2_new = list(
    df_mb.loc[df_mb["MB_flag_final"], ["time", "height"]]
         .itertuples(index=False, name=None)
)

plot_th_pairs(pairs_st2_new, "st2")
#%%

print(pairs_st2)
print(pairs_st2_new)
#%%

#
#
# -------------- BLOCCO II: significant coherence
#
#
#











#%%

#
#
# -------------- BLOCCO II: significant coherence
#
#
#
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def add_step2_coherence(df, date2fp, fband=(0.2, 0.4), win=0.03, coh_thresh=0.2):
    """
    Calcola la lower-bound coherence Cup tra u e p per gli eventi MB_flag_final=True.
    """
    df = df.copy()
    coh_vals, coh_pass = [], []

    for i, row in df.iterrows():
        if not row['MB_flag_final']:
            coh_vals.append(np.nan)
            coh_pass.append(False)
            continue

        t, h = row['time'], int(row['height'])
        day = pd.to_datetime(t).strftime('%Y-%m-%d')
        fp = date2fp.get(day, None)
        if fp is None:
            coh_vals.append(np.nan)
            coh_pass.append(False)
            continue

        # --- carica spettro dal pickle ---
        with open(fp, "rb") as f:
            d = pickle.load(f)
        spec = d["coherence"]

        fHz = spec["freq"].values
        Spp = spec["sp"].sel(time=np.datetime64(t), heights=h, method="nearest").values
        Suu = spec["su"].sel(time=np.datetime64(t), heights=h, method="nearest").values
        Cup = spec["cup"].sel(time=np.datetime64(t), heights=h, method="nearest").values

        # trova picco nello spettro di pressione
        mask_band = (fHz >= fband[0]) & (fHz <= fband[1])
        if not mask_band.any():
            coh_vals.append(np.nan)
            coh_pass.append(False)
            continue
        fstar = fHz[mask_band][np.argmax(Spp[mask_band])]

        # finestra attorno a f*
        mask_win = (fHz >= fstar - win) & (fHz <= fstar + win)
        if not mask_win.any():
            coh_vals.append(np.nan)
            coh_pass.append(False)
            continue

        # lower-bound coherence
        gamma2 = (Cup[mask_win]**2) / (Suu[mask_win] * Spp[mask_win])
        coh_pk = np.nanmax(gamma2)

        coh_vals.append(coh_pk)
        coh_pass.append(coh_pk >= coh_thresh)

    df["MB_step2_coh_value"] = coh_vals
    df["MB_step2_coh_pass"] = coh_pass
    return df

def plot_step2_examples(df_mb, date2fp, k=6, seed=0):
    """
    Plotta esempi di lower-bound coherence per eventi che passano Step-2.
    """
    rng = np.random.default_rng(seed)
    df_pass = df_mb.loc[df_mb["MB_step2_coh_pass"]].copy()
    if df_pass.empty:
        print("Nessun evento ha passato Step-2.")
        return

    pick_idx = rng.choice(df_pass.index.values, size=min(k, len(df_pass)), replace=False)
    df_pick = df_pass.loc[pick_idx].sort_values(["time", "height"])

    ncols = 3
    nrows = int(np.ceil(len(df_pick) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for ax, (_, row) in zip(axes.ravel(), df_pick.iterrows()):
        t, h = row["time"], int(row["height"])
        day = pd.to_datetime(t).strftime("%Y-%m-%d")
        fp = date2fp.get(day, None)
        if fp is None:
            ax.set_visible(False)
            continue

        with open(fp, "rb") as f:
            d = pickle.load(f)
        spec = d["spectra"]

        fHz = spec["freq"].values
        Spp = spec["sp"].sel(time=np.datetime64(t), heights=h, method="nearest").values
        Suu = spec["su"].sel(time=np.datetime64(t), heights=h, method="nearest").values
        Cup = spec["cup"].sel(time=np.datetime64(t), heights=h, method="nearest").values

        gamma2 = (Cup**2) / (Suu * Spp)

        ax.plot(fHz, gamma2, lw=1.5)
        ax.axhline(0.08, color="k", ls="--", lw=1)
        ax.axvspan(0.1, 0.4, alpha=0.15, color="grey")

        ax.set_xlim(0.05, 0.5)
        ax.set_ylim(0, 0.5)
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel("γ²_up (lb)")
        ax.set_title(f"{day}  t={pd.to_datetime(t).strftime('%H:%M')}  h={h}\n"
                     f"z={row['MB_z']:.2f}, coh_pk={row['MB_step2_coh_value']:.2f}")

    for ax in axes.ravel()[len(df_pick):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()

# aggiorna df_mb con Step-2
df_mb = add_step2_coherence(df_mb, date2fp)

# quanti passano?
print("Eventi Step-2 passati:", df_mb["MB_step2_coh_pass"].sum())

# plot esempi
plot_step2_examples(df_mb, date2fp, k=6)
#%%
gamma2 = np.abs(Sup)**2 / (Suu * Spp)

#%%
with open(fp, "rb") as f:
    d = pickle.load(f)
print(d['coherence'])


#%%
print(d['c'])