#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 13:28:45 2025

@author: mauro_ghirardelli
"""


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

import matplotlib.pyplot as plt
import numpy as np
import pickle

# Funzione per plottare tutte le curve cumulativamente
def plot_all_return_to_isotropy(xb_cumulative, yb_cumulative, title="All Return to Isotropy Paths"):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Triangolo baricentrico (1C, 2C, 3C)
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
    ax.set_title(title)

    # Traccia tutte le curve cumulativamente
    for xb, yb in zip(xb_cumulative, yb_cumulative):
        ax.plot(xb, yb, linewidth=1.0, alpha=0.6)  # Alza o abbassa alpha per visibilità

    plt.tight_layout()
    plt.show()

# -------------------------------------------
# Carica i file .pkl dal gruppo 2 (stazione 6)
xb_cumulative = []
yb_cumulative = []

# Itera su tutti i file in `file_paths_group2`
for file_path in file_paths_group2:
    with open(file_path, 'rb') as f:
        ds = pickle.load(f)
        
        # Assicurati che l'anisotropia smussata sia presente nel dataset
        if 'anisotropy_smooth' in ds:
            # Itera su tutte le dimensioni di time e height
            for time_idx in range(len(ds['time'])):
                for height_idx in range(len(ds['heights'])):
                    xb_cumulative.append(ds['anisotropy_smooth']['xb'].isel(time=time_idx, heights=height_idx).values)
                    yb_cumulative.append(ds['anisotropy_smooth']['yb'].isel(time=time_idx, heights=height_idx).values)

# Controlla quante curve sono state caricate
print(f"🎯 Totale curve caricate: {len(xb_cumulative)}")

# Plotta tutte le curve
plot_all_return_to_isotropy(xb_cumulative, yb_cumulative)


