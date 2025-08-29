#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 13:45:45 2025

@author: mauro_ghirardelli
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 08:49:01 2025

@author: mauro_ghirardelli
"""

import pickle, random
from pathlib import Path
import matplotlib.pyplot as plt
import sys

#import local libraries
sys.path.append('/Users/mauro_ghirardelli/Documents/windpy3/src/')
from spectral_analysis import Savitzky_Golay_log
from ogive import compute_ogive, reverse_ogive_reynolds_stress
#%%

# cartella principale che contiene st1/, st2/, …
base_dir = Path("/Users/mauro_ghirardelli/Documents/TEAMx/20250416_analisi/st6")

fp1 = f"{base_dir}/2025-02-01_10min.pkl"




    
with open(fp1, "rb") as f:
    ds1 = pickle.load(f)
    
    
#%%

print(ds1)


#%%
ds = ds1
# ── scegli il punto (ti, hi) che vuoi plottare ──────────────────────────
ti = 20       # indice tempo
hi = 0          # indice altezza
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

    norm = var_dict[var]

    

    
    # PROCESSED
    freq = ds['spectra']['freq'].values
    S = ds['spectra'][var].isel(time=ti, heights=hi).values
    y = freq * S / norm
    
    
    
    # Plot
 

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

    
    # PROCESSED
    f_pro = ds['spectra']['freq'].values
    C_pro = ds['spectra'][var].isel(time=ti, heights=hi).values



    # plot
 

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


f_smo  = ds['ogive']['freq_cutoff'].values

# ── helper per estrarre una componente (uu, vv, ww) dai tre gruppi ─────
def get_three( field ):  # field in {'uu','vv','ww'}

    y_smo = ds['ogive'][field].isel(time=ti, heights=hi).values
    return  norm_ogive(y_bin), norm_ogive(y_smo)

components = ['uu','vv','ww']
titles = [r"Ogiva $u'u'$", r"Ogiva $v'v'$", r"Ogiva $w'w'$"]

# ── PLOT: 3 pannelli (uu, vv, ww) ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

for ax, comp, title in zip(axes, components, titles):
    y_bin, y_smo = get_three(comp)



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

# ========= Parametri =========
NORMALIZE = False          # True -> ogive / ogive[0]
USE_ABS   = False          # True -> plottare |ogiva| per le covarianze
SHADE_BAND = (0.1, 0.6)    # banda da evidenziare in Hz

def norm_ogive(y):
    y = np.asarray(y, float)
    if NORMALIZE and np.isfinite(y[0]) and y[0] != 0:
        return y / y[0]
    return y

def maybe_abs(y):
    return np.abs(y) if USE_ABS else y

# ========= Sorgenti ogive disponibili =========
# Priorità: se esistono raw/smoothed le plottiamo entrambe; altrimenti usiamo ds['ogive']
sources = []
if 'ogive_raw' in ds and 'ogive_smoothed' in ds:
    sources = [
        ('Raw',       ds['ogive_raw']),
        ('Smoothed',  ds['ogive_smoothed']),
    ]
elif 'ogive' in ds:
    sources = [
        ('Ogive',     ds['ogive']),
    ]
else:
    raise KeyError("Non trovo né ds['ogive'] né (ds['ogive_raw'], ds['ogive_smoothed']).")

# Prendiamo la frequenza dal primo dataset disponibile
f = sources[0][1]['freq_cutoff'].values

# ========= 3 pannelli: uu, vv, ww =========
components = ['uu','vv','ww']
titles = [r"Ogiva $u'u'$", r"Ogiva $v'v'$", r"Ogiva $w'w'$"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

for ax, comp, title in zip(axes, components, titles):
    for label, og in sources:
        y = og[comp].isel(time=ti, heights=hi).values
        y = norm_ogive(y)
        ax.loglog(f, y, lw=2 if label.lower().startswith('smooth') else 1.5,
                  ls='--' if label.lower().startswith('smooth') else '-',
                  alpha=0.9 if len(sources)==1 else 0.9,
                  label=label)
    ax.axvspan(SHADE_BAND[0], SHADE_BAND[1], color='red', alpha=0.18, label="0.1–0.6 Hz")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlabel("Frequenza [Hz]")

axes[0].set_ylabel("Ogiva normalizzata" if NORMALIZE else "Ogiva")
handles, labels = axes[0].get_legend_handles_labels()
# Evita doppio label della banda se più sorgenti
handles2, labels2 = [], []
seen = set()
for h, l in zip(handles, labels):
    if l not in seen:
        handles2.append(h); labels2.append(l); seen.add(l)
axes[0].legend(handles2, labels2, loc="best")

fig.suptitle(f"Reverse Ogive – t={ti}, h={hi}", fontsize=13)
plt.tight_layout()
plt.show()


# ========= 3 pannelli: uv, uw, vw =========
components_cov = ['uv','uw','vw']
titles_cov = [r"Ogiva $u'v'$", r"Ogiva $u'w'$", r"Ogiva $v'w'$"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

for ax, comp, title in zip(axes, components_cov, titles_cov):
    for label, og in sources:
        y = og[comp].isel(time=ti, heights=hi).values
        y = maybe_abs(y)       # solo per visualizzazione; l’ogiva resta calcolata con segno
        y = norm_ogive(y)
        ax.semilogx(f, y, lw=2 if label.lower().startswith('smooth') else 1.5,
                  ls='--' if label.lower().startswith('smooth') else '-',
                  alpha=0.9 if len(sources)==1 else 0.9,
                  label=label)
    ax.axvspan(SHADE_BAND[0], SHADE_BAND[1], color='red', alpha=0.18, label="0.1–0.6 Hz")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlabel("Frequenza [Hz]")

ylabel = ("Ogiva normalizzata (modulo)" if NORMALIZE else "Ogiva (modulo)") if USE_ABS \
         else ("Ogiva normalizzata" if NORMALIZE else "Ogiva")
axes[0].set_ylabel(ylabel)

handles, labels = axes[0].get_legend_handles_labels()
handles2, labels2 = [], []
seen = set()
for h, l in zip(handles, labels):
    if l not in seen:
        handles2.append(h); labels2.append(l); seen.add(l)
axes[0].legend(handles2, labels2, loc="best")

fig.suptitle(f"Reverse Ogive – Covarianze (t={ti}, h={hi})", fontsize=13)
plt.tight_layout()
plt.show()
