# ================================================================
# Anisotropy Utilities
# ================================================================
# This module applies an anisotropy analysis to ogive-based Reynolds
# components. It includes:
#   - Anisotropy(...)   : colleague-provided routine (kept as-is)
#   - anisotropy_from_ogives(...) : adapter over ('time','heights','freq_cutoff')
#
# Author: <your name>
# Created: <date>
# ================================================================

# ================================================================
# Pointwise Anisotropy via xarray.apply_ufunc (element-wise)
# ================================================================

# import standard libraries
import numpy as np
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)

# import third part libraries

# import local libraries
import xarray as xr




def anisotropy_barycentric_ds(ogive):
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


# ------ helper
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




