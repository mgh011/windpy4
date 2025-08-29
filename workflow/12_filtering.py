#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 11:08:26 2025

@author: mauro_ghirardelli
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 08:33:20 2025

@author: mauro_ghirardelli
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 30 08:37:04 2025

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
    I_threshold = 0.5,
    #zL_abs_max=0.05
    require_fm_below_micro_min=True
)

print(f"Total Files: {len(file_paths_st2)}")
print(f"Total Pairs: {len(pairs_st2)}")   

plot_th_pairs(pairs_st2, "st2")
#%%

print(ds)
#%%

import pickle
import numpy as np
import matplotlib.pyplot as plt

# helper: test di appartenenza al triangolo (bordi inclusi)
_A = np.array([0.0, 0.0])                  # 2C
_C = np.array([0.5, np.sqrt(3)/2])         # 3C
_E = np.array([1.0, 0.0])                  # 1C
def _points_in_triangle(x, y):
    x1, y1 = _A; x2, y2 = _C; x3, y3 = _E
    c1 = (x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)
    c2 = (x3 - x2)*(y - y2) - (y3 - y2)*(x - x2)
    c3 = (x1 - x3)*(y - y3) - (y1 - y3)*(x - x3)
    return ((c1 <= 0) & (c2 <= 0) & (c3 <= 0)) | ((c1 >= 0) & (c2 >= 0) & (c3 >= 0))

def plot_pairs_density_on_triangle(file_paths, pairs, fmax=None, require_inside=False,
                                   gridsize=70, cmap="Greys"):
    """
    Densità cumulativa (hexbin in log) di tutte le curve (xb,yb) per le coppie in 'pairs'
    trovate nei file. Se fmax è dato, usa solo i punti con freq_cutoff <= fmax.
    Se require_inside=True, include SOLO le curve che restano interamente nel triangolo.
    """
    ax = plot_barycentric_triangle(perc=0.7, fill_zones=True)

    Xs, Ys = [], []
    for fp in file_paths:
        with open(fp, "rb") as f:
            d = pickle.load(f)
        ani = d["anisotropy_smooth"]

        times_arr   = ani["time"].values.astype("datetime64[ns]")
        heights_arr = ani["heights"].values

        for (t, h) in pairs:
            it = np.where(times_arr == np.datetime64(t, "ns"))[0]
            ih = np.where(heights_arr == int(h))[0]
            if it.size == 0 or ih.size == 0:
                continue

            ds_sel = ani.isel(time=int(it[0]), heights=int(ih[0]))
            xb = np.asarray(ds_sel["xb"].values)
            yb = np.asarray(ds_sel["yb"].values)
            f  = np.asarray(ds_sel["freq_cutoff"].values)

            if xb.size == 0:
                continue

            if fmax is not None:
                m = f <= float(fmax)
                if not np.any(m):
                    continue
                xb, yb = xb[m], yb[m]

            if require_inside:
                inside = _points_in_triangle(xb, yb)
                if not np.all(inside):
                    continue  # scarta curve che escono dal triangolo

            Xs.append(xb); Ys.append(yb)

    if not Xs:
        raise RuntimeError("Nessuna curva valida trovata per il plot di densità.")

    all_x = np.concatenate(Xs)
    all_y = np.concatenate(Ys)

    hb = ax.hexbin(all_x, all_y, gridsize=gridsize, cmap=cmap, bins='log', mincnt=1, linewidths=0.0)
    cb = plt.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label("log(count)")

    ttl = "Densità cumulativa"
    if fmax is not None:
        ttl += f" (f ≤ {fmax} Hz)"
    if require_inside:
        ttl += " — solo curve interne"
    ax.set_title(ttl)

    plt.tight_layout()
    plt.show()


# Solo f ≤ 5 Hz e scarta curve che escono dal triangolo
plot_pairs_density_on_triangle(file_paths_st2, pairs_st2, fmax=5.0, require_inside=True)
#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- helper: appartenenza al triangolo principale (bordi inclusi) ---
_A = np.array([0.0, 0.0])                  # 2C
_C = np.array([0.5, np.sqrt(3)/2])         # 3C
_E = np.array([1.0, 0.0])                  # 1C

def _points_in_triangle(x, y):
    x1, y1 = _A; x2, y2 = _C; x3, y3 = _E
    c1 = (x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)
    c2 = (x3 - x2)*(y - y2) - (y3 - y2)*(x - x2)
    c3 = (x1 - x3)*(y - y3) - (y1 - y3)*(x - x3)
    return ((c1 <= 0) & (c2 <= 0) & (c3 <= 0)) | ((c1 >= 0) & (c2 >= 0) & (c3 >= 0))

# --- helper: sottozone asintotiche + rettangolo in basso (per classificare il PUNTO DI PARTENZA) ---
def _triangles(state, perc=0.7):
    bary = np.array([0.5, np.sqrt(3)/6])
    A = np.array([0, 0])
    B = np.array([0.5*np.cos(np.pi/180*60), 0.5*np.sin(np.pi/180*60)])  # (0.25, ~0.433)
    C = np.array([0.5, np.sqrt(3)/2])
    D = np.array([1 - B[0], B[1]])
    E = np.array([1, 0])
    F = np.array([0.5, 0])
    if state == '1c':
        bary2 = [1 - bary[0]*perc, bary[1]*perc]
        t1 = np.array([E, [1 - B[0]*perc, D[1]*perc], bary2])
        t2 = np.array([E, F*(2 - perc), bary2])
    elif state == '2c':
        bary2 = bary * perc
        t1 = np.array([A, B * perc, bary2])
        t2 = np.array([A, F * perc, bary2])
    elif state == '3c':
        bary2 = [bary[0], C[1] - (C[1] - bary[1]) * perc]
        t1 = np.array([C, B * (2 - perc), bary2])
        t2 = np.array([C, [1 - B[0] * (2 - perc), D[1] * (2 - perc)], bary2])
    else:
        raise ValueError("state must be '1c','2c','3c'")
    return t1, t2

def _isin_triangle(pt, tri):
    x, y = pt
    (x1,y1),(x2,y2),(x3,y3) = tri
    c1 = (x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)
    c2 = (x3 - x2)*(y - y2) - (y3 - y2)*(x - x2)
    c3 = (x1 - x3)*(y - y3) - (y1 - y3)*(x - x3)
    return ((c1 < 0) and (c2 < 0) and (c3 < 0)) or ((c1 > 0) and (c2 > 0) and (c3 > 0))

def _start_in_zone(x0, y0, perc=0.7, h_ratio=0.2):
    """Ritorna '1c', '2c_axi' o '2c' in base al punto di partenza (x0,y0)."""
    # 1C?
    t1, t2 = _triangles('1c', perc)
    if _isin_triangle((x0,y0), t1) or _isin_triangle((x0,y0), t2):
        return '1c'
    # 2C?
    t1, t2 = _triangles('2c', perc)
    if _isin_triangle((x0,y0), t1) or _isin_triangle((x0,y0), t2):
        return '2c'
    # rettangolo in basso (axi)
    H = np.sqrt(3)/2
    if (0 <= x0 <= 1) and (0 <= y0 <= h_ratio*H):
        return '2c_axi'
    return None  # non classificato in questi tre bucket

def plot_pairs_density_by_start_zones(file_paths, pairs, fmax=5.0, perc=0.7, h_ratio=0.2,
                                      require_inside=True, gridsize=70, cmap="Greys"):
    """
    Tre densità (hexbin in log) in sottopannelli:
      - start 1C (basso-destra), start 2C (basso-sinistra), start rettangolo basso (axi).
    Di default include SOLO curve intere dentro il triangolo per f ≤ fmax.
    """
    # accumulatori per ciascun bucket
    buckets = {'1c': {'X':[], 'Y':[]}, '2c': {'X':[], 'Y':[]}, '2c_axi': {'X':[], 'Y':[]}}
    kept = {'1c':0, '2c':0, '2c_axi':0}
    skipped_any_out = 0
    skipped_no_f = 0
    skipped_empty = 0

    for fp in file_paths:
        with open(fp, "rb") as f:
            d = pickle.load(f)
        ani = d["anisotropy_smooth"]

        times_arr   = ani["time"].values.astype("datetime64[ns]")
        heights_arr = ani["heights"].values

        for (t, h) in pairs:
            it = np.where(times_arr == np.datetime64(t, "ns"))[0]
            ih = np.where(heights_arr == int(h))[0]
            if it.size == 0 or ih.size == 0:
                continue

            ds_sel = ani.isel(time=int(it[0]), heights=int(ih[0]))
            xb = np.asarray(ds_sel["xb"].values)
            yb = np.asarray(ds_sel["yb"].values)
            f  = np.asarray(ds_sel["freq_cutoff"].values)

            if xb.size == 0:
                skipped_empty += 1
                continue

            m = f <= float(fmax)
            if not np.any(m):
                skipped_no_f += 1
                continue

            x_sub, y_sub = xb[m], yb[m]

            if require_inside:
                if not np.all(_points_in_triangle(x_sub, y_sub)):
                    skipped_any_out += 1
                    continue

            # classificazione sul PUNTO DI PARTENZA (prima frequenza del sotto-insieme)
            x0, y0 = x_sub[0], y_sub[0]
            zone = _start_in_zone(x0, y0, perc=perc, h_ratio=h_ratio)
            if zone is None:
                continue

            buckets[zone]['X'].append(x_sub)
            buckets[zone]['Y'].append(y_sub)
            kept[zone] += 1

    # --- Plot: tre sottografi affiancati ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    titles = {'1c': 'Start: 1C (basso-destra)',
              '2c': 'Start: 2C (basso-sinistra)',
              '2c_axi': f'Start: fascia bassa (h≤{h_ratio}·H)'}
    order = ['1c', '2c', '2c_axi']

    any_plotted = False
    for ax, key in zip(axs, order):
        plot_barycentric_triangle(ax=ax, perc=perc, fill_zones=True)
        if buckets[key]['X']:
            all_x = np.concatenate(buckets[key]['X'])
            all_y = np.concatenate(buckets[key]['Y'])
            hb = ax.hexbin(all_x, all_y, gridsize=gridsize, cmap=cmap,
                           bins='log', mincnt=1, linewidths=0.0)
            any_plotted = True
            ax.set_title(f"{titles[key]} — kept={kept[key]}")
        else:
            ax.set_title(f"{titles[key]} — kept=0")

    if not any_plotted:
        raise RuntimeError("Nessuna curva valida per i tre bucket con i criteri scelti.")

    # una sola colorbar condivisa (opzionale)
    # prendi l'ultimo hexbin creato, se presente
    for a in axs:
        for coll in a.collections:
            hb = coll
    if 'hb' in locals():
        cbar = fig.colorbar(hb, ax=axs, shrink=0.9)
        cbar.set_label("log(count)")

    # report veloce
    print(f"Kept: 1C={kept['1c']}, 2C={kept['2c']}, AXI={kept['2c_axi']} | "
          f"Skipped: empty={skipped_empty}, no_f≤{fmax}={skipped_no_f}, out_of_triangle={skipped_any_out}")

    plt.show()

# di default: solo curve interamente interne al triangolo, f ≤ 5 Hz
plot_pairs_density_by_start_zones(
    file_paths=file_paths_st2,
    pairs=pairs_st2,
    fmax=5.0,          # soglia frequenze
    perc=0.7,          # dimensione sottozone 1C/2C
    h_ratio=0.2,       # altezza fascia bassa (rettangolo)
    require_inside=True,
    gridsize=70,
    cmap="Greys"
)

#%%

def plot_pairs_density_by_start_zones(
    file_paths, pairs, fmax=5.0, perc=0.7, h_ratio=0.2,
    require_inside=True, gridsize=70, cmap="Greys",
    show_median=True, micro_range=(0.1, 0.4)
):
    """
    Tre densità (hexbin log) in sottopannelli:
      - start 1C (basso-destra), start 2C (basso-sinistra), start rettangolo basso (axi).
    SOLO curve interamente dentro il triangolo per f ≤ fmax (se require_inside=True).
    Se show_median=True, sovrappone la MEDIANA del path; la banda micro_range è in rosso.
    """
    # accumulatori per ciascun bucket
    buckets = {
        '1c': {'X':[], 'Y':[], 'F':[]},
        '2c': {'X':[], 'Y':[], 'F':[]},
        '2c_axi': {'X':[], 'Y':[], 'F':[]}
    }
    kept = {'1c':0, '2c':0, '2c_axi':0}
    skipped_any_out = skipped_no_f = skipped_empty = 0

    for fp in file_paths:
        with open(fp, "rb") as f:
            d = pickle.load(f)
        ani = d["anisotropy_smooth"]

        times_arr   = ani["time"].values.astype("datetime64[ns]")
        heights_arr = ani["heights"].values

        for (t, h) in pairs:
            it = np.where(times_arr == np.datetime64(t, "ns"))[0]
            ih = np.where(heights_arr == int(h))[0]
            if it.size == 0 or ih.size == 0:
                continue

            ds_sel = ani.isel(time=int(it[0]), heights=int(ih[0]))
            xb = np.asarray(ds_sel["xb"].values)
            yb = np.asarray(ds_sel["yb"].values)
            f  = np.asarray(ds_sel["freq_cutoff"].values)

            if xb.size == 0:
                skipped_empty += 1
                continue

            m = f <= float(fmax)
            if not np.any(m):
                skipped_no_f += 1
                continue

            x_sub, y_sub, f_sub = xb[m], yb[m], f[m]

            if require_inside and not np.all(_points_in_triangle(x_sub, y_sub)):
                skipped_any_out += 1
                continue

            # classificazione sul PUNTO DI PARTENZA del sottoinsieme
            x0, y0 = x_sub[0], y_sub[0]
            zone = _start_in_zone(x0, y0, perc=perc, h_ratio=h_ratio)
            if zone is None:
                continue

            buckets[zone]['X'].append(x_sub)
            buckets[zone]['Y'].append(y_sub)
            buckets[zone]['F'].append(f_sub)
            kept[zone] += 1

    # helper per plottare segmenti colorati contigui
    def _plot_masked_segments(ax, x, y, mask, **kwargs):
        idx = np.where(mask)[0]
        if idx.size < 2:
            return
        cuts = np.where(np.diff(idx) > 1)[0] + 1
        for seg in np.split(idx, cuts):
            if seg.size > 1:
                ax.plot(x[seg], y[seg], **kwargs)

    # --- Plot: tre sottografi ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    titles = {'1c': 'Start: 1C (basso-destra)',
              '2c': 'Start: 2C (basso-sinistra)',
              '2c_axi': f'Start: fascia bassa (h≤{h_ratio}·H)'}
    order = ['1c', '2c', '2c_axi']

    any_plotted = False
    last_hb = None

    for ax, key in zip(axs, order):
        plot_barycentric_triangle(ax=ax, perc=perc, fill_zones=True)

        Xs, Ys, Fs = buckets[key]['X'], buckets[key]['Y'], buckets[key]['F']
        if Xs:
            all_x = np.concatenate(Xs)
            all_y = np.concatenate(Ys)
            hb = ax.hexbin(all_x, all_y, gridsize=gridsize, cmap=cmap,
                           bins='log', mincnt=1, linewidths=0.0)
            last_hb = hb
            any_plotted = True

            if show_median:
                # griglia comune = prima curva del bucket
                f_grid = Fs[0]
                X_interp, Y_interp = [], []
                for x_arr, y_arr, f_arr in zip(Xs, Ys, Fs):
                    xi = np.full_like(f_grid, np.nan, dtype=float)
                    yi = np.full_like(f_grid, np.nan, dtype=float)
                    valid = (f_grid >= f_arr[0]) & (f_grid <= f_arr[-1])
                    if np.any(valid):
                        xi[valid] = np.interp(f_grid[valid], f_arr, x_arr)
                        yi[valid] = np.interp(f_grid[valid], f_arr, y_arr)
                    X_interp.append(xi); Y_interp.append(yi)

                X_stack = np.stack(X_interp, axis=0)  # (n_curves, n_freq)
                Y_stack = np.stack(Y_interp, axis=0)

                x_med = np.nanmedian(X_stack, axis=0)
                y_med = np.nanmedian(Y_stack, axis=0)

                ok = ~np.isnan(x_med) & ~np.isnan(y_med)
                if np.any(ok):
                    f_ok = f_grid[ok]
                    x_ok = x_med[ok]
                    y_ok = y_med[ok]

                    # maschera microbarom
                    fmin_mb, fmax_mb = micro_range
                    m_mb = (f_ok >= fmin_mb) & (f_ok <= fmax_mb)

                    # fuori banda (blu) e dentro banda (rosso)
                    _plot_masked_segments(ax, x_ok, y_ok, ~m_mb, lw=2.2, color='tab:blue', zorder=3)
                    _plot_masked_segments(ax, x_ok, y_ok,  m_mb, lw=2.2, color='tab:red',  zorder=4)

                    # legende dummy per non ripetere etichette
                    ax.plot([], [], lw=2.2, color='tab:blue')
                    ax.plot([], [], lw=2.2, color='tab:red',  label='Microbarom Range')

            ax.set_title(f"{titles[key]} — kept={kept[key]}")
        else:
            ax.set_title(f"{titles[key]} — kept=0")

        ax.legend(loc="lower left", frameon=False)

    if not any_plotted:
        raise RuntimeError("Nessuna curva valida per i tre bucket con i criteri scelti.")

    if last_hb is not None:
        cbar = fig.colorbar(last_hb, ax=axs, shrink=0.9)
        cbar.set_label("log(count)")

    print(f"Kept: 1C={kept['1c']}, 2C={kept['2c']}, AXI={kept['2c_axi']} | "
          f"Skipped: empty={skipped_empty}, no_f≤{fmax}={skipped_no_f}, out_of_triangle={skipped_any_out}")

    plt.show()


plot_pairs_density_by_start_zones(
    file_paths=file_paths_st2,
    pairs=pairs_st2,
    fmax=5.0,
    perc=0.7,
    h_ratio=0.2,
    require_inside=True,
    gridsize=70,
    cmap="Greys",
    show_median=True,
    micro_range=(0.1, 0.4)   # <— banda colorata in rosso
)

#%%
