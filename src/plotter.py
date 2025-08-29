#import standar libaries
import os
import pickle
import warnings

#import third part libraries
import numpy as np
import matplotlib.pyplot as plt


#import local libraries


# ----------------------


def pdf_pp_f(
    file_paths,
    N_bins=50,
    title="PDF of Spectral Amplitude (sp)",
    cmap='viridis',
    output_filepath=None,
    qc_threshold=None,
    norm=True 
):
    """
    Compute and plot a 2D PDF of f vs f * Epp (normalized or not) from multiple pickled datasets.

    Parameters
    ----------
    file_paths : list of str
        List of pickle files, each with 'spectra' and 'stats' datasets.
    N_bins : int
        Number of log-spaced bins on each axis.
    title : str
        Title for the plot.
    cmap : str
        Colormap to use.
    output_filepath : str or None
        If given, save the figure to this path.
    qc_threshold : float or None
        Skip time slices with QCnan > threshold.
    norm : bool
        If True, normalize Epp by sigma_pp^2. If False, use raw Epp.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    X_list, A_list = [], []

    for path in file_paths:
        if not os.path.isfile(path):
            warnings.warn(f"File not found: {path}")
            continue

        with open(path, 'rb') as f:
            ds = pickle.load(f)

        spec_ds = ds['spectra']
        stat_ds = ds['stats']

        freqs = spec_ds.freq.values
        hi = 0  # Only first height

        for ti in range(spec_ds.time.size):
            qc = stat_ds.isel(time=ti, heights=hi).QCnan.values
            if qc_threshold is not None and qc > qc_threshold:
                continue

            E = spec_ds.isel(time=ti, heights=hi)['sp'].values.real

            if norm:
                sigma2 = stat_ds.isel(time=ti, heights=hi)['pp'].values
                E = E / sigma2  # normalizzato

            X = freqs
            A = X * E  # prodotto f * E_pp o f * E_pp / sigma_pp^2

            mask = (X > 0) & (A > 0) & np.isfinite(A)
            X_list.append(X[mask])
            A_list.append(A[mask])

    # Final arrays
    X_flat = np.concatenate(X_list)
    A_flat = np.concatenate(A_list)

    # Bin edges
    bins_X = np.logspace(np.log10(X_flat.min()), np.log10(X_flat.max()), N_bins + 1)
    bins_A = np.logspace(np.log10(A_flat.min()), np.log10(A_flat.max()), N_bins + 1)

    # 2D histogram
    H, xedges, yedges = np.histogram2d(X_flat, A_flat, bins=[bins_X, bins_A])

    # Normalize per X bin
    Hn = np.zeros_like(H)
    row = H.sum(axis=1)
    nz = row > 0
    Hn[nz, :] = (H[nz, :].T / row[nz]).T * 100

    # Plot
    Xg, Ag = np.meshgrid(xedges, yedges)
    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(Xg, Ag, Hn.T, shading='auto', cmap=cmap, vmin=0, vmax=10)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("f [Hz]")

    if norm:
        ax.set_ylabel(r'$f \cdot E_{pp} / \sigma_{pp}^2$')
    else:
        ax.set_ylabel(r'$f \cdot E_{pp}$')

    ax.set_title(title)

    # Microbarom band
    #ax.axvline(0.12, linestyle='--', color='k')
    #ax.axvline(0.4, linestyle='--', color='k')

    # Colorbar with fixed ticks
    cbar = plt.colorbar(pcm, ax=ax, ticks=[0, 5, 10, 15, 20])
    cbar.set_label("Percent per f-bin")

    ax.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()


    if output_filepath:
        fig.savefig(output_filepath, dpi=150)

    return fig
    

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



