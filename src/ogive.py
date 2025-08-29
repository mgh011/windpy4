#import standar libaries
import numpy as np
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)

#import third part libraries
import numpy as np
import xarray as xr
from scipy.interpolate import PchipInterpolator

#import local libraries

# Main routine
# -------------


def compute_ogive(spectra, n_points: int = None):
    """
    Compute ogives over the full frequency range per (time,height).
    Se n_points è valorizzato, l'ogiva viene riscampionata log-spaced a n_points.
    """
    results = []

    times = spectra.time.values
    heights = spectra.heights.values

    loop_iter = [(ti, hi) for ti in range(len(times)) for hi in range(len(heights))]
    loop_iter = tqdm(loop_iter, desc="Ogive calculation", leave=True)

    for ti, hi in loop_iter:
        t_val = times[ti]
        h_val = heights[hi]

        f_vals, og_uu, og_vv, og_ww, og_uv, og_uw, og_vw = reverse_ogive_reynolds_stress(
            spectra, time_index=ti, height_index=hi,
        )

        ds_out = xr.Dataset(
            data_vars={
                'uu': (['time', 'heights', 'freq_cutoff'], og_uu[np.newaxis, np.newaxis, :]),
                'vv': (['time', 'heights', 'freq_cutoff'], og_vv[np.newaxis, np.newaxis, :]),
                'ww': (['time', 'heights', 'freq_cutoff'], og_ww[np.newaxis, np.newaxis, :]),
                'uv': (['time', 'heights', 'freq_cutoff'], og_uv[np.newaxis, np.newaxis, :]),
                'uw': (['time', 'heights', 'freq_cutoff'], og_uw[np.newaxis, np.newaxis, :]),
                'vw': (['time', 'heights', 'freq_cutoff'], og_vw[np.newaxis, np.newaxis, :]),
            },
            coords={'freq_cutoff': f_vals, 'time': [t_val], 'heights': [h_val]},
        )

        # Downsample opzionale (log-spaced, shape-preserving)
        if n_points is not None:
            ds_out = downsample_ogive_log(ds_out, n_points=n_points, cast_float32=True)

        results.append(ds_out)

    return xr.combine_by_coords(results)







# Subroutines
# -------------

def reverse_ogive_reynolds_stress(ds, time_index=0, height_index=0):
    """
    Reverse ogives (∫_{fmin}^∞ S(f) df) con stessa lunghezza della freq.
    NaN-safe e area-preserving via somma cumulativa dei contributi S*Δf.
    """
    freqs = ds['su'].coords['freq'].values.astype(float)
    # assicurati che le freq siano crescenti
    order = np.argsort(freqs)
    f = freqs[order]

    def _sel(name):
        return np.asarray(ds[name].isel(time=time_index, heights=height_index).values, float)[order]

    su  = _sel('su')
    sv  = _sel('sv')
    sw  = _sel('sw')
    cuv = _sel('cuv')
    cuw = _sel('cuw')
    cvw = _sel('cvw')

    # Δf con prepend (stessa lunghezza)
    df = np.diff(f, prepend=f[0])

    # sostituisci NaN con 0 nei contributi d'area (ignora buchi)
    def rev_cumint(S):
        A = np.nan_to_num(S, nan=0.0) * df
        # reverse cumulative sum, stessa lunghezza di f
        return np.cumsum(A[::-1])[::-1]

    og_uu = rev_cumint(su)
    og_vv = rev_cumint(sv)
    og_ww = rev_cumint(sw)
    og_uv = rev_cumint(cuv)
    og_uw = rev_cumint(cuw)
    og_vw = rev_cumint(cvw)

    # frequenza "cutoff" = f stessa (N punti), non N-1
    return f, og_uu, og_vv, og_ww, og_uv, og_uw, og_vw

# --- helper: risampling log-spaced dell'ogiva, senza medie nei bin ----------

def downsample_ogive_log(ds_ogive: xr.Dataset, n_points: int = 3000, cast_float32: bool = True) -> xr.Dataset:
    """
    Downsample delle ogive lungo 'freq_cutoff' su griglia log-spaced di n_points.
    Usa PCHIP (shape-preserving), nessuna extrapolazione (fuori range -> NaN).
    Mantiene dims (time, heights); riduce solo 'freq_cutoff'.
    """
    assert "freq_cutoff" in ds_ogive.coords, "Manca coord 'freq_cutoff' nel dataset di ogive"
    f = ds_ogive["freq_cutoff"].values.astype(float)
    fmin, fmax = float(np.nanmin(f)), float(np.nanmax(f))
    f_tgt = np.logspace(np.log10(fmin), np.log10(fmax), int(n_points))

    has_h = "heights" in ds_ogive.dims
    T = ds_ogive.sizes.get("time", 1)
    H = ds_ogive.sizes.get("heights", 1)

    out_vars = {}
    for name, da in ds_ogive.data_vars.items():
        arr = np.asarray(da.values, float)  # (time, heights, freq_cutoff) oppure (time, freq_cutoff)
        if has_h:
            out = np.full((T, H, f_tgt.size), np.nan, dtype=float)
            for ti in range(T):
                for hi in range(H):
                    y = arr[ti, hi, :]
                    m = np.isfinite(f) & np.isfinite(y)
                    if m.sum() < 2:
                        continue
                    p = PchipInterpolator(f[m], y[m], extrapolate=False)
                    out[ti, hi, :] = p(f_tgt)
            dims = ("time", "heights", "freq_cutoff")
        else:
            out = np.full((T, f_tgt.size), np.nan, dtype=float)
            for ti in range(T):
                y = arr[ti, :]
                m = np.isfinite(f) & np.isfinite(y)
                if m.sum() < 2:
                    continue
                p = PchipInterpolator(f[m], y[m], extrapolate=False)
                out[ti, :] = p(f_tgt)
            dims = ("time", "freq_cutoff")

        if cast_float32:
            out = out.astype(np.float32)
        out_vars[name] = (dims, out)

    coords = {k: ds_ogive.coords[k] for k in ds_ogive.coords if k != "freq_cutoff"}
    coords["freq_cutoff"] = ("freq_cutoff", f_tgt.astype(np.float32) if cast_float32 else f_tgt)
    return xr.Dataset(out_vars, coords=coords)

