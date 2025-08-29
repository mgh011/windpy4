#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 10:28:27 2025

@author: mauro_ghirardelli
"""

import pickle, random
from pathlib import Path

# cartella principale che contiene st1/, st2/, …
base_dir = Path("/Users/mauro_ghirardelli/Documents/TEAMx/20250416_analisi/st6")

# cerca tutti i .pkl in tutte le sottocartelle delle stazioni
all_pickle_paths = list(base_dir.glob("**/*.pkl"))


if not all_pickle_paths:
    raise FileNotFoundError("Nessun *.pkl trovato sotto base_dir!")
# scegline uno a caso
pick_path = random.choice(all_pickle_paths)
print(f"🗄  Apro il file: {pick_path}")

with open(pick_path, "rb") as f:
    ds = pickle.load(f)

# mostra il contenuto xarray (o qualunque cosa ci sia dentro)
print(ds)

#%%
import matplotlib.pyplot as plt


# ── scegli il punto (ti, hi) che vuoi plottare ──────────────────────────
ti = 63         # indice tempo
hi = 0          # indice altezza


# ── estraI lo spettro S_u(f) per quel (t,h) ─────────────────────────────
freq = ds['spectra']['freq'].values                         # 1-D (nfreq,)
S_u  = ds['spectra']['su'].isel(time=ti, heights=hi).values # 1-D (nfreq,)
S_v  = ds['spectra']['sv'].isel(time=ti, heights=hi).values # 1-D (nfreq,)
S_w  = ds['spectra']['sw'].isel(time=ti, heights=hi).values # 1-D (nfreq,)
S_p  = ds['spectra']['sp'].isel(time=ti, heights=hi).values # 1-D (nfreq,)


# statistica di riferimento (es. varianza uu) per normalizzare
uu_var = ds['stats']['uu'].isel(time=ti, heights=hi).values # scalare
vv_var = ds['stats']['vv'].isel(time=ti, heights=hi).values # scalare
ww_var = ds['stats']['ww'].isel(time=ti, heights=hi).values # scalare
pp_var = ds['stats']['pp'].isel(time=ti, heights=hi).values # scalare


# normalizzazione tipica ogiva: f·S_u(f) / uu_var  (esempio)
yu = freq * S_u / uu_var
yv = freq * S_v / vv_var
yw = freq * S_w / ww_var
yp = freq * S_p / pp_var


# ── plot log-log ─────────────────────────────────────────────────────────
plt.figure(figsize=(6,4))
plt.loglog(freq, yu)
plt.loglog(freq, yv)
plt.loglog(freq, yw)
plt.loglog(freq, yp)
plt.xlabel('Frequency [Hz]')
plt.ylabel('su, sv, sw')
plt.title(f'Spettro –  t index={ti},  h index={hi}')
plt.grid(True, which='both', ls='--', alpha=0.3)
plt.show()

#%%
print(ds['ogive'])
#%%

import matplotlib.pyplot as plt
import numpy as np

# ── scegli il punto (ti, hi) che vuoi plottare ──────────────────────────

# ── estrai la frequenza e le ogive dal dataset ──────────────────────────
freq = ds['ogive']['freq_cutoff'].values

og_uu = ds['ogive']['uu'].isel(time=ti, heights=hi).values
og_vv = ds['ogive']['vv'].isel(time=ti, heights=hi).values
og_ww = ds['ogive']['ww'].isel(time=ti, heights=hi).values
og_uv = ds['ogive']['uv'].isel(time=ti, heights=hi).values
og_uw = ds['ogive']['uw'].isel(time=ti, heights=hi).values
og_vw = ds['ogive']['vw'].isel(time=ti, heights=hi).values

# ── PLOT 1: Reverse ogive normalizzate – VARIANZE ───────────────────────
plt.figure(figsize=(6, 4))
"""
plt.loglog(freq, og_uu / og_uu[0], label="u'u'")
plt.loglog(freq, og_vv / og_vv[0], label="v'v'")
plt.loglog(freq, og_ww / og_ww[0], label="w'w'")
"""

plt.loglog(freq, og_uu, label="u'u'")
plt.loglog(freq, og_vv, label="v'v'")
plt.loglog(freq, og_ww, label="w'w'")

plt.axvspan(0.1, 0.6, color='red', alpha=0.3, label="0.1–0.6 Hz")

plt.xlabel("Frequenza [Hz]")
plt.ylabel("Ogiva normalizzata")
plt.title(f"Reverse Ogive Normalizzate – Varianze (t={ti}, h={hi})")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.show()

# ── PLOT 2: Reverse ogive normalizzate – COVARIANZE ─────────────────────
plt.figure(figsize=(8, 5))
plt.loglog(freq, np.abs(og_uv), label="u'v'")
plt.loglog(freq, np.abs(og_uw), label="u'w'")
plt.loglog(freq, np.abs(og_vw), label="v'w'")

plt.axvspan(0.1, 0.6, color='red', alpha=0.3, label="0.1–0.6 Hz")

plt.xlabel("Frequenza [Hz]")
plt.ylabel("Ogiva normalizzata (modulo)")
plt.title(f"Reverse Ogive Normalizzate – Covarianze (t={ti}, h={hi})")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.show()


#%%
import matplotlib.pyplot as plt
import numpy as np

# ── scegli il punto (ti, hi) che vuoi plottare ──────────────────────────

# ── estrai la frequenza e le ogive dal dataset ──────────────────────────
freq = ds['ogive']['freq_cutoff'].values

og_uu = ds['ogive']['uu'].isel(time=ti, heights=hi).values
og_vv = ds['ogive']['vv'].isel(time=ti, heights=hi).values
og_ww = ds['ogive']['ww'].isel(time=ti, heights=hi).values
og_uv = ds['ogive']['uv'].isel(time=ti, heights=hi).values
og_uw = ds['ogive']['uw'].isel(time=ti, heights=hi).values
og_vw = ds['ogive']['vw'].isel(time=ti, heights=hi).values

# ── PLOT 1: Reverse ogive normalizzate – VARIANZE ───────────────────────
plt.figure(figsize=(6, 4))
"""
plt.loglog(freq, og_uu / og_uu[0], label="u'u'")
plt.loglog(freq, og_vv / og_vv[0], label="v'v'")
plt.loglog(freq, og_ww / og_ww[0], label="w'w'")
"""

plt.semilogx(freq, og_uu, label="u'u'")
plt.semilogx(freq, og_vv, label="v'v'")
plt.semilogx(freq, og_ww, label="w'w'")
plt.semilogx(freq, og_uv, label="u'v'")
plt.semilogx(freq, og_uw, label="u'w'")
plt.semilogx(freq, og_vw, label="v'w'")



plt.axvspan(0.1, 0.6, color='red', alpha=0.3, label="0.1–0.6 Hz")

plt.xlabel("Frequenza [Hz]")
plt.ylabel("Ogiva normalizzata")
plt.title(f"Reverse Ogive Normalizzate – Varianze (t={ti}, h={hi})")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.show()










#%%

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def validate_ogive_for_var(spectra_ds: xr.Dataset, ogive_ds: xr.Dataset,
                           var_name: str = "uu", time_idx: int = 0, height_idx: int = 0,
                           xscale: str = "log"):
    """
    Confronta spettro originale S(f) e ogiva Og(f_min) per una variabile (uu/vv/..).
    Fa 3 controlli: integrale totale, derivata numerica, e plot comparativi.

    Parameters
    ----------
    spectra_ds : xr.Dataset
        Dataset originale con spettri/co-spettri (su,sv,sw,cuv,cuw,cvw) e coord 'freq'.
        Mappatura var_name -> nome nello spettro:
            uu<-su, vv<-sv, ww<-sw, uv<-cuv, uw<-cuw, vw<-cvw
    ogive_ds : xr.Dataset
        Output di compute_ogive con variabili (uu,vv,ww,uv,uw,vw) e coord 'freq_cutoff'.
    var_name : str
        Uno tra 'uu','vv','ww','uv','uw','vw'.
    time_idx, height_idx : int
        Indici del punto da analizzare.
    xscale : {'linear','log'}
        Scala per l’asse frequenza.

    Returns
    -------
    dict con errori/metriche.
    """
    # mappatura nome ogiva -> nome spettro
    spec_map = {"uu":"su", "vv":"sv", "ww":"sw", "uv":"cuv", "uw":"cuw", "vw":"cvw"}
    if var_name not in spec_map:
        raise ValueError("var_name must be one of uu,vv,ww,uv,uw,vw")

    # estrai frequenze e dati
    f = spectra_ds["freq"].values
    S = spectra_ds[spec_map[var_name]].isel(time=time_idx, heights=height_idx).values
    fmin = ogive_ds["freq_cutoff"].values
    Og = ogive_ds[var_name].isel(time=time_idx, heights=height_idx).values

    # 1) integrale totale vs ogiva al primo cutoff
    integ_total = np.trapz(S, f)
    og_low = Og[0]  # assumendo freq_cutoff crescente, primo = cutoff più basso
    abs_err_total = float(og_low - integ_total)
    rel_err_total = float(abs_err_total / (np.abs(integ_total)+1e-15))

    # 2) derivata numerica dOg/dfmin ~ -S(f)
    # vogliamo confrontare -dOg/dfmin a S, ma S è campionato su f, mentre dOg su fmin.
    # Se fmin e f coincidono (stesso array), il confronto è diretto.
    if len(fmin) != len(f) or np.max(np.abs(fmin - f[:len(fmin)])) > 1e-12:
        # riporta S su la griglia di fmin (interpolazione semplice)
        S_on_fmin = np.interp(fmin, f, S)
    else:
        S_on_fmin = S[:len(fmin)]

    # derivata numerica all’indietro (stessa lunghezza di fmin, con primo punto via forward)
    dOg = np.gradient(Og, fmin)  # dOg/dfmin
    S_rec = -dOg  # dovrebbe ~ S_on_fmin

    # metriche di aderenza (escludi estremi se rumorosi)
    i0, i1 = 1, len(fmin)-2 if len(fmin) > 3 else len(fmin)-1
    if i1 <= i0:  # pochi punti, fallback a tutti
        i0, i1 = 0, len(fmin)
    num = S_on_fmin[i0:i1]
    rec = S_rec[i0:i1]
    # MAPE robusta (evita divisioni per 0 usando denominatore “+ small”)
    mape = float(np.mean(np.abs(rec - num) / (np.abs(num) + 1e-12)))
    # correlazione di Pearson
    if rec.size > 1:
        corr = float(np.corrcoef(rec, num)[0,1])
    else:
        corr = np.nan

    # --- plot ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    # (A) spettro e cumulativa (ogiva)
    axs[0].plot(f, S, label=f"Spectrum {var_name.upper()}")
    ax2 = axs[0].twinx()
    ax2.plot(fmin, Og, color='tab:orange', label="Ogive (cumulative)")
    axs[0].set_xscale(xscale); ax2.set_xscale(xscale)
    axs[0].set_xlabel("Frequency f")
    axs[0].set_ylabel("S(f)")
    ax2.set_ylabel("Ogive(f_min) = ∫_{f_min}^{f_max} S(f) df")
    axs[0].grid(True, which='both', axis='both', alpha=0.3)
    # legende combinate
    lines1, labels1 = axs[0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs[0].legend(lines1+lines2, labels1+labels2, loc="best")
    axs[0].set_title(f"Ogive vs Spectrum @ time={time_idx}, height={height_idx}")

    # (B) -dOg/df vs S (sulla stessa griglia)
    axs[1].plot(fmin, S_on_fmin, label="S(f) (on f_min grid)")
    axs[1].plot(fmin, S_rec, linestyle='--', label="- dOg/df_min (recovered)")
    axs[1].set_xscale(xscale)
    axs[1].set_xlabel("Lower cutoff frequency f_min")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True, which='both', axis='both', alpha=0.3)
    axs[1].legend(loc="best")
    axs[1].set_title(f"Derivative Check (corr={corr:.3f}, MAPE={100*mape:.1f}%)")

    plt.tight_layout()

    return {
        "integral_total": float(integ_total),
        "ogive_at_lowest_cutoff": float(og_low),
        "abs_error_total": abs_err_total,
        "rel_error_total": rel_err_total,
        "corr_-dOg_df_vs_S": corr,
        "mape_-dOg_df_vs_S": mape
    }

#%%
print(ds)

#%%

# supponiamo:
# spectra_ds = results["spectra"]    # contiene su,sv,sw,cuv,cuw,cvw con coord 'freq'
# ogive_ds   = results["ogive"]      # output di compute_ogive

report = validate_ogive_for_var(ds['spectra'], ds['ogive'], var_name="uu", time_idx=0, height_idx=0)
print(report)

# ripeti per 'uv' ecc. cambiando var_name

