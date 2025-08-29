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
from filtering import find_th_pairs, plot_th_pairs
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
)

print(f"Total Files: {len(file_paths_st2)}")
print(f"Total Pairs: {len(pairs_st2)}")   

plot_th_pairs(pairs_st2, "st2")


#%%


# lista di tutte le coppie valide trovate
valid_ds = []   # conterrà tuple (fp, t, h, ds_sel)

for fp in file_paths_st2:
    with open(fp, "rb") as f:
        d = pickle.load(f)

    ani = d["anisotropy_smooth"]

    times_arr   = ani["time"].values.astype("datetime64[ns]")
    heights_arr = ani["heights"].values

    for (t, h) in pairs_st2:
        it = np.where(times_arr == np.datetime64(t, "ns"))[0]
        ih = np.where(heights_arr == int(h))[0]

        if it.size > 0 and ih.size > 0:
            it, ih = int(it[0]), int(ih[0])
            ds_sel = ani.isel(time=it, heights=ih)
            valid_ds.append((fp, times_arr[it], int(heights_arr[ih]), ds_sel))

# ora valid_ds[0] è la prima coppia valida, valid_ds[1] la seconda, ecc.
print(f"Trovate {len(valid_ds)} coppie valide in totale.")

# esempio: prendi la seconda coppia
fp2, t2, h2, ds_test2 = valid_ds[1]
print("Seconda coppia valida:")
print("file:", fp2)
print("time:", t2, "height:", h2)
print(ds_test2)
#%%
for key in d['stats'].keys():
    print(key)
#%%



def plot_barycentric_triangle(ax=None, perc=0.7, fill_zones=True):
    """
    Plot the Lumley barycentric triangle with optional asymptotic zones (1C, 2C, 3C).

    Parameters
    ----------
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates a new one.
    perc : float
        Fraction used for asymptotic zone triangles (default=0.7).
    fill_zones : bool
        If True, fill asymptotic subzones (1C, 2C, 3C).
    """
    # main triangle points
    A = np.array([0, 0])
    C = np.array([0.5, np.sqrt(3)/2])
    E = np.array([1, 0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    # plot main triangle
    tri = np.array([A, C, E, A])
    ax.plot(tri[:,0], tri[:,1], 'k-', lw=1.5)

    # optional fill of asymptotic zones
    if fill_zones:
        from matplotlib.patches import Polygon
        # reuse your triangles() function
        def triangles(state, perc):
            bary = np.array([0.5, np.sqrt(3)/6])
            A = np.array([0, 0])
            B = np.array([0.5*np.cos(np.pi/180*60), 0.5*np.sin(np.pi/180*60)])
            C = np.array([0.5, np.sqrt(3)/2])
            D = np.array([1 - B[0], B[1]])
            E = np.array([1, 0])
            F = np.array([0.5, 0])
            if state == '1c':
                bary2 = [1 - bary[0]*perc, bary[1]*perc]
                triangle1 = [E, [1 - B[0]*perc, D[1]*perc], bary2]
                triangle2 = [E, F*(2 - perc), bary2]
            elif state == '2c':
                bary2 = bary * perc
                triangle1 = [A, B * perc, bary2]
                triangle2 = [A, F * perc, bary2]
            elif state == '3c':
                bary2 = [bary[0], C[1] - (C[1] - bary[1]) * perc]
                triangle1 = [C, B * (2 - perc), bary2]
                triangle2 = [C, [1 - B[0] * (2 - perc), D[1] * (2 - perc)], bary2]
            return np.array(triangle1), np.array(triangle2)

        colors = {"1c":"#ff9999","2c":"#99ccff","3c":"#99ff99"}
        for state in ["1c","2c","3c"]:
            t1,t2 = triangles(state, perc)
            poly1 = Polygon(t1, closed=True, facecolor=colors[state], alpha=0.3)
            poly2 = Polygon(t2, closed=True, facecolor=colors[state], alpha=0.3)
            ax.add_patch(poly1)
            ax.add_patch(poly2)

    ax.set_aspect("equal")
    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,np.sqrt(3)/2+0.1)
    ax.set_xlabel("$B_x$")
    ax.set_ylabel("$B_y$")
    ax.set_title("Barycentric triangle (with asymptotic zones)")

    return ax


ax = plot_barycentric_triangle(perc=0.7, fill_zones=True)


fp2, t2, h2, ds_test2 = valid_ds[70]

xb = np.asarray(ds_test2["xb"].values)
yb = np.asarray(ds_test2["yb"].values)
f  = np.asarray(ds_test2["freq_cutoff"].values)

# filtro: solo frequenze <= 5 Hz
mask = f <= 5.0
xb = xb[mask]
yb = yb[mask]
f  = f[mask]

# maschera microbarom 0.1–0.4 Hz
mask_micro = (f >= 0.1) & (f <= 0.4)

# punti fuori microbarom (blu)
ax.scatter(xb[~mask_micro], yb[~mask_micro], c="blue", s=10)

# punti dentro microbarom (rosso)
ax.scatter(xb[mask_micro], yb[mask_micro], c="red", s=15, label="microbarom range")

# start/end evidenziati
ax.plot(xb[0],  yb[0], "o", ms=8, mfc="none", mec="k")
ax.plot(xb[-1], yb[-1], "s", ms=8, mfc="none", mec="k")

ax.legend(loc="lower left", frameon=False)
plt.tight_layout()
plt.show()
#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1) Raccogli tutte le coppie valide ----------
def build_valid_pairs(file_paths, pairs):
    """
    Ritorna una lista di tuple (file_path, time, height, ds_sel)
    dove ds_sel = d['anisotropy_smooth'].isel(time=it, heights=ih)
    """
    found = []
    for fp in file_paths:
        with open(fp, "rb") as f:
            d = pickle.load(f)
        ani = d["anisotropy_smooth"]

        times_arr   = ani["time"].values.astype("datetime64[ns]")
        heights_arr = ani["heights"].values

        for (t, h) in pairs:
            it = np.where(times_arr == np.datetime64(t, "ns"))[0]
            ih = np.where(heights_arr == int(h))[0]
            if it.size > 0 and ih.size > 0:
                it, ih = int(it[0]), int(ih[0])
                ds_sel = ani.isel(time=it, heights=ih)
                found.append((fp, times_arr[it], int(heights_arr[ih]), ds_sel))
    return found

# ---------- 2) Selettore flessibile ----------
def select_pair(valid_list, idx=None, time=None, height=None):
    """
    Scegli una coppia da valid_list:
      - per indice: idx (0-based)
      - per (time, height): passare entrambi
    Ritorna (fp, t, h, ds_sel).
    """
    if idx is not None:
        if not (0 <= idx < len(valid_list)):
            raise IndexError(f"idx {idx} fuori range (len={len(valid_list)})")
        return valid_list[idx]

    if (time is not None) and (height is not None):
        # match esatto
        for fp, t, h, ds_sel in valid_list:
            if np.datetime64(t, "ns") == np.datetime64(time, "ns") and int(h) == int(height):
                return (fp, t, h, ds_sel)
        raise ValueError("Nessuna coppia trovata per il (time,height) richiesto.")

    raise ValueError("Specificare idx oppure (time & height).")

# ---------- 3) Plot: solo punti, microbarom rosso ----------
def plot_return_to_isotropy_points(ds_sel, title=None, fmax=5.0, micro_range=(0.1, 0.4)):
    """
    Plotta (xb,yb) come punti sul triangolo:
      - rosso nella banda microbarom (micro_range)
      - blu altrove
      - evidenzia start/end
    """
    ax = plot_barycentric_triangle(perc=0., fill_zones=True)

    xb = np.asarray(ds_sel["xb"].values)
    yb = np.asarray(ds_sel["yb"].values)
    f  = np.asarray(ds_sel["freq_cutoff"].values)

    # filtro frequenze
    mask_f = f <= float(fmax)
    xb, yb, f = xb[mask_f], yb[mask_f], f[mask_f]

    # microbarom mask
    fmin, fmax_mb = micro_range
    mask_micro = (f >= fmin) & (f <= fmax_mb)

    # scatter
    ax.scatter(xb[~mask_micro], yb[~mask_micro], s=10, label="out-band")
    ax.scatter(xb[mask_micro],  yb[mask_micro],  s=15, label=f"{fmin}-{fmax_mb} Hz")

    # start / end
    ax.plot(xb[0],  yb[0],  "o", ms=8, mfc="none", mec="k", label=f"start {f[0]:.3f} Hz")
    ax.plot(xb[-1], yb[-1], "s", ms=8, mfc="none", mec="k", label=f"end {f[-1]:.3f} Hz")

    if title:
        ax.set_title(title)

    ax.legend(loc="lower left", frameon=False)
    plt.tight_layout()
    plt.show()

# ---------- USO ----------
# 1) costruisci la lista una volta
valid_ds = build_valid_pairs(file_paths_st2, pairs_st2)
print(f"Trovate {len(valid_ds)} coppie valide.")

# 2a) scegli per indice (esempio: seconda coppia -> idx=1)
fp, t, h, ds_choice = select_pair(valid_ds, idx=1)
print("Selezionata:", fp, t, h)

# 2b) oppure scegli per (time, height)
# fp, t, h, ds_choice = select_pair(valid_ds, time=np.datetime64("2025-01-23T12:10:00"), height=2)

# 3) plot
plot_return_to_isotropy_points(
    ds_choice,
    title=f"{os.path.basename(fp)} | t={np.datetime_as_string(t)} h={h}",
    fmax=5.0,
    micro_range=(0.1, 0.4)
)

#%%
def plot_cumulative_first_points(valid_list):
    """
    Plotta sul triangolo il PRIMO punto (xb[0], yb[0]) 
    di tutte le coppie valide (valid_ds).
    """
    ax = plot_barycentric_triangle(perc=0.7, fill_zones=True)

    all_x = []
    all_y = []

    for fp, t, h, ds_sel in valid_list:
        xb = np.asarray(ds_sel["xb"].values)
        yb = np.asarray(ds_sel["yb"].values)
        if xb.size > 0:
            all_x.append(xb[0])
            all_y.append(yb[0])

    if len(all_x) == 0:
        raise RuntimeError("Nessuna coppia valida con valori disponibili.")

    ax.scatter(all_x, all_y, c="blue", s=20, alpha=0.6, label="first point of each pair")

    ax.legend(loc="lower left", frameon=False)
    plt.tight_layout()
    plt.show()

# --- USO ---
plot_cumulative_first_points(valid_ds)
#%%
def plot_cumulative_last_points(valid_list, fmax=5.0):
    """
    Plotta sul triangolo l'ULTIMO punto (xb[k], yb[k]) per ogni coppia valida,
    dove k è l'ultimo indice tale che freq_cutoff[k] <= fmax.
    """
    ax = plot_barycentric_triangle(perc=0.7, fill_zones=True)

    xs, ys = [], []
    skipped = 0

    for fp, t, h, ds_sel in valid_list:
        xb = np.asarray(ds_sel["xb"].values)
        yb = np.asarray(ds_sel["yb"].values)
        f  = np.asarray(ds_sel["freq_cutoff"].values)

        if xb.size == 0:
            skipped += 1
            continue

        mask = f <= float(fmax)
        if not np.any(mask):
            skipped += 1
            continue

        # ultimo indice valido sotto la soglia
        k = np.where(mask)[0][-1]
        xs.append(xb[k])
        ys.append(yb[k])

    if len(xs) == 0:
        raise RuntimeError("Nessuna coppia con frequenze <= fmax trovata.")

    ax.scatter(xs, ys, c="green", s=20, alpha=0.6, label=f"last point (f ≤ {fmax} Hz)")
    ax.legend(loc="lower left", frameon=False)
    plt.tight_layout()
    plt.show()

    if skipped:
        print(f"Saltate {skipped} coppie senza punti con f ≤ {fmax} Hz.")

plot_cumulative_last_points(valid_ds, fmax=5)
#%%
import numpy as np
import matplotlib.pyplot as plt

# --- helper: test di appartenenza al triangolo (vettoriale) ---
_A = np.array([0.0, 0.0])                  # 2C
_C = np.array([0.5, np.sqrt(3)/2])         # 3C
_E = np.array([1.0, 0.0])                  # 1C

def _points_in_triangle(x, y):
    x1, y1 = _A; x2, y2 = _C; x3, y3 = _E
    c1 = (x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)
    c2 = (x3 - x2)*(y - y2) - (y3 - y2)*(x - x2)
    c3 = (x1 - x3)*(y - y3) - (y1 - y3)*(x - x3)
    return ((c1 <= 0) & (c2 <= 0) & (c3 <= 0)) | ((c1 >= 0) & (c2 >= 0) & (c3 >= 0))

def plot_cumulative_density_inside(valid_list, fmax=5.0, gridsize=70, cmap="Greys"):
    """
    Densità cumulativa dei punti (xb,yb) prendendo SOLO le coppie la cui curva
    per tutte le frequenze <= fmax resta interamente dentro il triangolo.
    """
    ax = plot_barycentric_triangle(perc=0.7, fill_zones=True)

    Xs, Ys = [], []
    kept = skipped_empty = skipped_over_f = skipped_out = 0

    for _, _, _, ds_sel in valid_list:
        xb = np.asarray(ds_sel["xb"].values)
        yb = np.asarray(ds_sel["yb"].values)
        f  = np.asarray(ds_sel["freq_cutoff"].values)

        if xb.size == 0:
            skipped_empty += 1
            continue

        m = f <= float(fmax)
        if not np.any(m):
            skipped_over_f += 1
            continue

        x_sub, y_sub = xb[m], yb[m]

        # escludi l'intera curva se QUALSIASI punto esce dal triangolo
        if not np.all(_points_in_triangle(x_sub, y_sub)):
            skipped_out += 1
            continue

        Xs.append(x_sub)
        Ys.append(y_sub)
        kept += 1

    if not Xs:
        raise RuntimeError("Nessuna curva interamente interna al triangolo per f ≤ fmax.")

    all_x = np.concatenate(Xs)
    all_y = np.concatenate(Ys)

    hb = ax.hexbin(all_x, all_y, gridsize=gridsize, cmap=cmap,
                   bins='log', mincnt=1, linewidths=0.0)
    cb = plt.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label("log(count)")

    ax.set_title(f"Densità (solo curve interne, f ≤ {fmax} Hz) — kept={kept}, "
                 f"skipped: empty={skipped_empty}, no_f≤{fmax}={skipped_over_f}, out={skipped_out}")
    plt.tight_layout()
    plt.show()

plot_cumulative_density_inside(valid_ds, fmax=5.0, gridsize=500, cmap="Greys")
