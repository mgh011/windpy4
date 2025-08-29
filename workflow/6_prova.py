
import pickle, random
from pathlib import Path
import matplotlib.pyplot as plt
import sys

#import local libraries
sys.path.append('/Users/mauro_ghirardelli/Documents/windpy3/src/')
from spectral_analysis import Savitzky_Golay_log
from ogive import compute_ogive, reverse_ogive_reynolds_stress


# --- 1) crea dataset smussato (stessa griglia di freq) -------------------
import numpy as np
import xarray as xr

def is_cospec(name: str) -> bool:
    name = name.lower()
    return any(k in name for k in ["cuv","cuw","cvw","suv","suw","svw","uv","uw","vw"]) and not name.startswith(("su","sv","sw","sp"))

def make_smoothed_from_raw(ds_raw, num_log_pts=2000, window_pts=21, polyorder=2):
    """Restituisce un Dataset con le stesse dims/coords di ds_raw ma smussato."""
    freq = ds_raw["freq"].values
    has_heights = "heights" in ds_raw.dims
    dims = ds_raw[ next(iter(ds_raw.data_vars)) ].dims  # es: ('time','heights','freq')
    out = {}
    for name, da in ds_raw.data_vars.items():
        arr = np.asarray(da.values, float)  # (time[,heights],freq)
        arr_s = np.full_like(arr, np.nan, dtype=float)
        it_time = range(da.sizes["time"])
        it_h = range(da.sizes.get("heights", 1))
        for ti in it_time:
            for hi in it_h:
                spec = arr[ti, hi, :] if has_heights else arr[ti, :]
                if np.sum(np.isfinite(spec)) < 4:
                    continue
                segmented = is_cospec(name)   # True per uv/uw/vw
                _, sm = Savitzky_Golay_log(freq, spec,
                                           num_log_pts=num_log_pts,
                                           window_pts=window_pts,
                                           polyorder=polyorder,
                                           segmented=segmented)
                if has_heights:
                    arr_s[ti, hi, :] = sm
                else:
                    arr_s[ti, :] = sm
        out[name] = (dims, arr_s)
    coords = {k: ds_raw.coords[k] for k in ds_raw.coords}
    return xr.Dataset(out, coords=coords)


#%%

# cartella principale che contiene st1/, st2/, …
base_dir = Path("/Users/mauro_ghirardelli/Documents/TEAMx/20250416_analisi/st6")

fp1 = f"{base_dir}/2025-02-01_10min.pkl"



with open(fp1, "rb") as f:
    ds1 = pickle.load(f)
    
# costruisci il "solo smoothed" partendo dall'originale
ds1_smooth = make_smoothed_from_raw(ds1["raw_spectra"],
                                    num_log_pts=2000,
                                    window_pts=21,
                                    polyorder=2)

# --- 2) ogive: raw vs smooth (stesso calcolo) ----------------------------
og_raw    = compute_ogive(ds1["raw_spectra"])
og_smooth = compute_ogive(ds1_smooth)


#%%
# --- 3) plot covarianze uv, uw, vw firmate (niente modulo) ---------------
import matplotlib.pyplot as plt

ti = 15
hi = 0
freq = ds1["raw_spectra"]["freq"].values

def pick(name_list, og_ds):
    for n in name_list:
        if n in og_ds: return n
    raise KeyError(f"manca {name_list}")
    
def align_for_plot(f, O):
    """Rende f e O della stessa lunghezza per il plot.
    - Se len(O) == len(f)-1 assume integrazione su intervalli → usa f[1:].
    - Se len(O) == len(f) usa f così com'è.
    - Altrimenti tronca al minimo comune.
    """
    nf, no = len(f), len(O)
    if no == nf - 1:
        return f[1:], O
    elif no == nf:
        return f, O
    else:
        n = min(nf, no)
        return f[:n], O[:n]



nm_uv = pick(["uv","cuv","suv"], og_raw)
nm_uw = pick(["uw","cuw","suw"], og_raw)
nm_vw = pick(["vw","cvw","svw"], og_raw)

pairs = [
    ("u'v'", og_raw[nm_uv].isel(time=ti, heights=hi).values,
             og_smooth[nm_uv].isel(time=ti, heights=hi).values),
    ("u'w'", og_raw[nm_uw].isel(time=ti, heights=hi).values,
             og_smooth[nm_uw].isel(time=ti, heights=hi).values),
    ("v'w'", og_raw[nm_vw].isel(time=ti, heights=hi).values,
             og_smooth[nm_vw].isel(time=ti, heights=hi).values)
]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, (lab, O_raw, O_smo) in zip(axes, pairs):
    x, y = align_for_plot(freq, O_raw)
    ax.semilogx(x, y, alpha=0.4, label="Raw")
    
    x, y = align_for_plot(freq, O_smo)
    ax.semilogx(x, y, lw=2, label="Smoothed")


    ax.set_title(f"Ogiva {lab}")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_xlabel("Frequency [Hz]")
axes[0].set_ylabel("Ogiva (∫ C(f) df)")
axes[0].legend()
plt.tight_layout()
plt.show()

# --- 4) diagnosi: la covarianza totale deve coincidere -------------------
def covariance_from_ogive(freq, og):
    # reverse-ogive → il limite a f→0 è la covarianza; prendi il primo valore finito
    og = np.asarray(og, float)
    first = np.nan
    for v in og:
        if np.isfinite(v):
            first = v; break
    return first

for nm, lab in [(nm_uv,"u'v'"), (nm_uw,"u'w'"), (nm_vw,"v'w'")]:
    O_raw = og_raw[nm].isel(time=ti, heights=hi).values
    O_smo = og_smooth[nm].isel(time=ti, heights=hi).values
    cov_raw = covariance_from_ogive(freq, O_raw)
    cov_smo = covariance_from_ogive(freq, O_smo)
    rel_err = (cov_smo - cov_raw)/cov_raw if cov_raw not in (0, np.nan) else np.nan
    print(f"{lab}: cov_raw={cov_raw:.6g}, cov_smooth={cov_smo:.6g}, rel.diff={rel_err:.2%}")
