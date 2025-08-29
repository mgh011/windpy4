

#import standar libaries
import os
import pickle
import warnings
from pathlib import Path
import sys

#import third part libraries
import numpy as np
import matplotlib.pyplot as plt


#import local libraries
sys.path.append('/Users/mauro_ghirardelli/Documents/windpy3/src/')
from plotter import pdf_pp_f



# Cartella principale
base_dir = Path("/Users/mauro_ghirardelli/Documents/TEAMx/20250416_analisi/")

# Definizione dei gruppi di stazioni
group1 = {"st1", "st2", "st3"}
group2 = {"st4", "st5", "st6"}

# Liste finali
file_paths_group1 = []
file_paths_group2 = []

# Cerca tutti i .pkl nei sottodirectory
all_pickle_paths = base_dir.glob("**/*.pkl")

for path in all_pickle_paths:
    if "10min" not in path.stem.lower():
        continue  # salta se non contiene "hf"

    # Trova il nome della cartella stazione (es: "st2") cercando tra i genitori
    for parent in path.parents:
        if parent.name in group1:
            file_paths_group1.append(path)
            break
        elif parent.name in group2:
            file_paths_group2.append(path)
            break

# Controllo finale
print(f"📁 File in group1 (st1-3): {len(file_paths_group1)}")
print(f"📁 File in group2 (st4-6): {len(file_paths_group2)}")

#%%
import json
import xarray as xr
station = 'st6'

# Load configuration file
config_path = '/Users/mauro_ghirardelli/Documents/windpy3/conf/config_10min.txt'
with open(config_path, 'r') as file:
    config = json.load(file)
    

file_path = f"{config['path']}{station}/2025-02-12_10min.pkl"


with open(file_path, 'rb') as f:
    ds = pickle.load(f)

print(ds)
#%%

#%%

import numpy as np
import xarray as xr

# funzione element-wise: da uu,vv,ww,uv,uw,vw -> (xb, yb)
def _anisotropy_xy(uu, vv, ww, uv, uw, vw):
    # trace e normalizzazione "sicura"
    trace = uu + vv + ww
    denom_ok = np.isfinite(trace) and (trace != 0.0)

    def norm(x):
        if denom_ok:
            return x / trace
        else:
            return 0.0  # se trace non valido, b = -I/3 -> (xb, yb) = (0, 0)

    # tensore di anisotropia b_ij simmetrico
    b11 = norm(uu) - 1.0/3.0
    b22 = norm(vv) - 1.0/3.0
    b33 = norm(ww) - 1.0/3.0
    b12 = norm(uv); b13 = norm(uw); b23 = norm(vw)

    b = np.array([[b11, b12, b13],
                  [b12, b22, b23],
                  [b13, b23, b33]], dtype=float)

    # eigendecomposizione per matrici simmetriche
    vals, vecs = np.linalg.eigh(b)        # ascendente
    vals = vals[::-1]                     # decrescente
    vals = vals - vals.mean()             # enforce trace=0 (robustezza)

    l1, l2, l3 = vals[0], vals[1], vals[2]

    # coordinate baricentriche (Pope / Emory-Jaccarino)
    xb = (l1 - l2) + 1.5*l3 + 0.5
    yb = (np.sqrt(3)/2.0) * (3.0*l3 + 1.0)
    return xb, yb

def anisotropy_barycentric_ds(ogive: xr.Dataset) -> xr.Dataset:
    xb, yb = xr.apply_ufunc(
        _anisotropy_xy,
        ogive['uu'], ogive['vv'], ogive['ww'],
        ogive['uv'], ogive['uw'], ogive['vw'],
        input_core_dims=[[], [], [], [], [], []],   # element-wise su tutte le altre dims
        output_core_dims=[[], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float],
    )
    xb.name = 'xb'
    yb.name = 'yb'
    return xr.Dataset({'xb': xb, 'yb': yb})

# --- uso ---
ogive_raw = ds['ogive_raw']
ogive_smooth = ds['ogive']
ds_bary_raw = anisotropy_barycentric_ds(ogive_raw)
ds_bary_smooth = anisotropy_barycentric_ds(ogive_smooth)
# ds_bary.dims -> ('time', 'heights', 'freq_cutoff')
# ds_bary.data_vars -> xb, yb
#%%
print(ds_bary_smooth)
#%%

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def plot_return_to_isotropy(
    ds_bary: xr.Dataset,
    time_idx = 0,
    height_idx = 0,
    fmax = 8.0,
    fmin = None,
    freq_step = 1,
    scatter_size = 35,
    show_path = True,
    annotate_ends = True,
    cmap = "viridis",
    cbar = True,
    ax  = None
):
    """
    Plotta il percorso 'return to isotropy' nel triangolo baricentrico usando ds_bary
    con variabili:
      - xb (time, heights, freq_cutoff)
      - yb (time, heights, freq_cutoff)
    Colori = freq_cutoff.
    """

    # --- selezione in frequenza
    if fmin is None:
        try:
            fmin = float(ds_bary.freq_cutoff.min())
        except Exception:
            fmin = None

    if fmin is None:
        ds_sel = ds_bary.sel(freq_cutoff=slice(None, fmax))
    else:
        ds_sel = ds_bary.sel(freq_cutoff=slice(fmin, fmax))

    xb = ds_sel['xb'].isel(time=time_idx, heights=height_idx).values
    yb = ds_sel['yb'].isel(time=time_idx, heights=height_idx).values
    f  = ds_sel['freq_cutoff'].values

    if xb.size == 0:
        raise ValueError("Nessun punto dopo il filtro in frequenza: controlla fmin/fmax.")

    # downsampling per chiarezza
    step = max(int(freq_step), 1)
    xb, yb, f = xb[::step], yb[::step], f[::step]

    # --- axes
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created = True
    ax.set_aspect('equal', adjustable='box')

    # triangolo baricentrico (1C, 2C, 3C)
    tri_x = np.array([1.0, 0.0, 0.5, 1.0])
    tri_y = np.array([0.0, 0.0, np.sqrt(3.0)/2.0, 0.0])
    ax.plot(tri_x, tri_y, linewidth=1.5, color="black")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, np.sqrt(3.0)/2.0 + 0.05)

    ax.text(1.0, 0.0, "1C", va='top', ha='left')
    ax.text(0.0, 0.0, "2C", va='top', ha='right')
    ax.text(0.5, np.sqrt(3.0)/2.0, "3C (isotropic)", va='bottom', ha='center')

    ax.set_xlabel("Barycentric $x_b$")
    ax.set_ylabel("Barycentric $y_b$")
    ax.set_title(f"Return to Isotropy (freq ≤ {fmax:g} Hz)")

    # path + scatter colorato per frequenza
    if show_path:
        ax.plot(xb, yb, linewidth=1.0, alpha=0.8)

    sc = ax.scatter(xb, yb, s=scatter_size, c=f, cmap=cmap)

    if cbar:
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("freq_cutoff [Hz]")

    # annotazioni inizio/fine
    if annotate_ends and xb.size >= 1:
        ax.annotate("low cutoff",  (xb[0],  yb[0]),  xytext=(5,  6), textcoords='offset points')
        ax.annotate("high cutoff", (xb[-1], yb[-1]), xytext=(5, -12), textcoords='offset points')
        ax.annotate("", xy=(xb[-1], yb[-1]), xytext=(xb[0], yb[0]),
                    arrowprops=dict(arrowstyle="->", lw=1))

    if created:
        plt.tight_layout()
    return ax

#%%

ds_bary_rawm = ds['anisotropy_raw']
ds_bary_smooth = ds['anisotropy_smooth']


#%%
_ = plot_return_to_isotropy(ds_bary_raw, time_idx=52, height_idx=0, fmax=8.0, freq_step=2)

_ = plot_return_to_isotropy(ds_bary_smooth, time_idx=52, height_idx=0, fmax=8.0, freq_step=2)
#%%

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# --- PATCH piccola: aggiungo `norm` e uso la stessa per scatter/colorbar
def plot_return_to_isotropy(
    ds_bary,
    time_idx = 0,
    height_idx = 0,
    fmax = 8.0,
    fmin = None,
    freq_step = 1,
    scatter_size = 35,
    show_path = True,
    annotate_ends = True,
    cmap = "viridis",
    cbar = True,
    ax  = None,
    norm =  None,   # <--- NEW
):
    # selezione in frequenza
    if fmin is None:
        try:
            fmin = float(ds_bary.freq_cutoff.min())
        except Exception:
            fmin = None
    ds_sel = (ds_bary.sel(freq_cutoff=slice(fmin, fmax))
              if fmin is not None else
              ds_bary.sel(freq_cutoff=slice(None, fmax)))

    xb = ds_sel['xb'].isel(time=time_idx, heights=height_idx).values
    yb = ds_sel['yb'].isel(time=time_idx, heights=height_idx).values
    f  = ds_sel['freq_cutoff'].values
    if xb.size == 0:
        raise ValueError("Nessun punto dopo il filtro in frequenza: controlla fmin/fmax.")

    step = max(int(freq_step), 1)
    xb, yb, f = xb[::step], yb[::step], f[::step]

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created = True
    ax.set_aspect('equal', adjustable='box')

    tri_x = np.array([1.0, 0.0, 0.5, 1.0])
    tri_y = np.array([0.0, 0.0, np.sqrt(3.0)/2.0, 0.0])
    ax.plot(tri_x, tri_y, linewidth=1.5, color="black")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, np.sqrt(3.0)/2.0 + 0.05)

    ax.text(1.0, 0.0, "1C", va='top', ha='left')
    ax.text(0.0, 0.0, "2C", va='top', ha='right')
    ax.text(0.5, np.sqrt(3.0)/2.0, "3C (isotropic)", va='bottom', ha='center')

    ax.set_xlabel("Barycentric $x_b$")
    ax.set_ylabel("Barycentric $y_b$")
    ax.set_title(f"Return to Isotropy (freq ≤ {fmax:g} Hz)")

    if show_path:
        ax.plot(xb, yb, linewidth=1.0, alpha=0.8)

    sc = ax.scatter(xb, yb, s=scatter_size, c=f, cmap=cmap, norm=norm)  # <--- usa norm

    if cbar:
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("freq_cutoff [Hz]")

    if annotate_ends and xb.size >= 1:
        ax.annotate("low cutoff",  (xb[0],  yb[0]),  xytext=(5,  6), textcoords='offset points')
        ax.annotate("high cutoff", (xb[-1], yb[-1]), xytext=(5, -12), textcoords='offset points')
        ax.annotate("", xy=(xb[-1], yb[-1]), xytext=(xb[0], yb[0]),
                    arrowprops=dict(arrowstyle="->", lw=1))

    if created:
        plt.tight_layout()
    return ax

# --- Nuova funzione: confronto RAW vs SMOOTH affiancati, colorbar condivisa
def plot_return_to_isotropy_compare(
    ds_bary_raw: xr.Dataset,
    ds_bary_smooth: xr.Dataset,
    time_idx=0,
    height_idx=0,
    fmax=8.0,
    fmin=None,
    freq_step=1,
    scatter_size=35,
    cmap="viridis",
):
    # fmin condiviso
    if fmin is None:
        fmin = float(min(ds_bary_raw.freq_cutoff.min(), ds_bary_smooth.freq_cutoff.min()))
    # stessa normalizzazione colore su [fmin, fmax]
    norm = Normalize(vmin=fmin, vmax=fmax)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # RAW
    plot_return_to_isotropy(
        ds_bary_raw, time_idx=time_idx, height_idx=height_idx,
        fmax=fmax, fmin=fmin, freq_step=freq_step,
        scatter_size=scatter_size, cmap=cmap, cbar=False,
        ax=axes[0], norm=norm
    )
    axes[0].set_title("Return to Isotropy — RAW")
    # SMOOTH
    plot_return_to_isotropy(
        ds_bary_smooth, time_idx=time_idx, height_idx=height_idx,
        fmax=fmax, fmin=fmin, freq_step=freq_step,
        scatter_size=scatter_size, cmap=cmap, cbar=False,
        ax=axes[1], norm=norm
    )
    axes[1].set_title("Return to Isotropy — SMOOTH")

    # Colorbar condivisa
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cb.set_label("freq_cutoff [Hz]")

    fig.tight_layout()
    return fig, axes

_ = plot_return_to_isotropy_compare(
        ds_bary_raw, ds_bary_smooth,
        time_idx=60, height_idx=0,
        fmax=5.0, freq_step=1
    )
#%%
print("Spaziatura media:", np.mean(np.diff(ds_bary_raw.freq_cutoff.values)))
print("Min:", ds_bary_raw.freq_cutoff.min().item(),
      "Max:", ds_bary_raw.freq_cutoff.max().item())

#%%
print(ds)