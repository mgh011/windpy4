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

#%%

pdf_pp_f(
    file_paths_group1,
    N_bins=200,
    title="PDF of Spectral Amplitude (sp)",
    cmap='gist_ncar_r',
    output_filepath=None,
    qc_threshold=0.3,
    norm= False
)

#%%

def compute_pdf_data_grouped(file_paths, norm=True, qc_threshold=None, N_bins=400):
    """Compute normalized 2D histogram and mesh for a group of files."""
    X_list, A_list = [], []

    for path in file_paths:
        if not os.path.isfile(path):
            continue

        with open(path, 'rb') as f:
            ds = pickle.load(f)

        spec_ds = ds['spectra']
        stat_ds = ds['stats']
        freqs = spec_ds.freq.values
        hi = 0

        for ti in range(spec_ds.time.size):
            qc = stat_ds.isel(time=ti, heights=hi).QCnan.values
            if qc_threshold is not None and qc > qc_threshold:
                continue
            
            """
            U_threshold = 0.5
            U = stat_ds.isel(time=ti, heights=hi).meanU.values
            if U < U_threshold:
                continue
            
            
            uw = stat_ds.isel(time=ti, heights=hi).uw.values
            if uw < 0:
                continue
            """
            vT = stat_ds.isel(time=ti, heights=hi).vT.values
            if vT < 0:
                continue
            wT = stat_ds.isel(time=ti, heights=hi).wT.values
            if wT > 0:
                continue

            E = spec_ds.isel(time=ti, heights=hi)['sp'].values.real
            if norm:
                sigma2 = stat_ds.isel(time=ti, heights=hi)['pp'].values
                E = E / sigma2

            X = freqs
            A = X * E

            mask = (X > 0) & (A > 0) & np.isfinite(A)
            X_list.append(X[mask])
            A_list.append(A[mask])

    X_flat = np.concatenate(X_list)
    A_flat = np.concatenate(A_list)

    bins_X = np.logspace(np.log10(X_flat.min()), np.log10(X_flat.max()), N_bins + 1)
    bins_A = np.logspace(np.log10(A_flat.min()), np.log10(A_flat.max()), N_bins + 1)

    H, _, _ = np.histogram2d(X_flat, A_flat, bins=[bins_X, bins_A])
    Hn = np.zeros_like(H)
    row = H.sum(axis=1)
    nz = row > 0
    Hn[nz, :] = (H[nz, :].T / row[nz]).T * 100

    Xg, Ag = np.meshgrid(bins_X, bins_A)
    return {"Hn": Hn, "Xg": Xg, "Ag": Ag}

# Calcolo una sola volta
data_group1 = compute_pdf_data_grouped(file_paths_group1, norm=False, qc_threshold=0.1)
data_group2 = compute_pdf_data_grouped(file_paths_group2, norm=False, qc_threshold=0.1)


#%%

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def plot_dual_pdf(data1, data2,
                  title1="a) Gruppo A", title2="b) Gruppo B",
                  vmax=10,
                  output_filepath=None,
                  colorbar_pos=[0.88, 0.15, 0.03, 0.7],
                  norm=True,
                  blend_white=20,
                  base_cmap_name="gist_rainbow_r"):
    """
    Plot stacked dual PDF with one custom-blended colormap and manual colorbar position.

    Parameters
    ----------
    data1, data2 : dict
        Output from compute_pdf_data_grouped().
    title1, title2 : str
        Internal captions for subplot 1 and 2.
    vmax : float
        Max value for color normalization.
    output_filepath : str or None
        Path to save figure.
    colorbar_pos : list of 4 floats
        [left, bottom, width, height] for manual colorbar position.
    norm : bool
        Whether the data is normalized (for ylabel).
    blend_white : int
        Number of colors at the low end to blend with white.
    base_cmap_name : str
        Name of the matplotlib colormap to use as base.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Create blended colormap
    orig = mpl.cm.get_cmap(base_cmap_name, 256)
    colors = orig(np.linspace(0, 1, 256))
    for i in range(blend_white):
        blend_factor = i / max(1, blend_white - 1)
        colors[i] = blend_factor * colors[i] + (1 - blend_factor) * np.array([1, 1, 1, 1])
    custom_cmap = mpl.colors.ListedColormap(colors, name=f"custom_{base_cmap_name}")

    # Plot setup
    fig, axs = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

    for ax, data, label in zip(axs, [data1, data2], [title1, title2]):
        pcm = ax.pcolormesh(data["Xg"], data["Ag"], data["Hn"].T,
                            shading='auto', cmap=custom_cmap, vmin=0, vmax=vmax)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-9, 1e-2)

        ylabel = r'$f \cdot E_{p} / \sigma_{p}^2$' if norm else r'$f \cdot E_{p} [Pa^2]$'
        ax.set_ylabel(ylabel)

        ax.grid(True, which='both', ls='--', alpha=0.5)

        # Internal caption
        ax.text(0.05, 0.9, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')

    axs[-1].set_xlabel("f [Hz]")

    # Custom-positioned colorbar
    cax = fig.add_axes(colorbar_pos)
    cbar = fig.colorbar(pcm, cax=cax, ticks=[0, vmax//2, vmax])
    cbar.set_label("Percent per f-bin")

    plt.tight_layout()

    if output_filepath:
        fig.savefig(output_filepath, dpi=150)

    return fig

plot_dual_pdf(
    data_group1, data_group2,
    vmax=5,
    norm=False,
    title1="st1–st3 vT<0",
    title2="st4–st6 vT<0",
    colorbar_pos=[0.96, 0.16, 0.025, 0.8],  # più larga e staccata
    blend_white=30,                        # sfumatura più graduale
    base_cmap_name="gist_rainbow_r"
)
