import os, json, pickle
import numpy as np
import xarray as xr

# Config
config_path = '/Users/mauro_ghirardelli/Documents/windpy3/conf/config_10min.txt'
with open(config_path, 'r') as f:
    config = json.load(f)

station = "st2"
pkl_path = f"{config['path']}{station}/2025-02-12_10min.pkl"

# 1) Leggi il pickle
with open(pkl_path, 'rb') as f:
    ds = pickle.load(f)  # dict con raw_spectra, spectra, coherence, ...

# 2) Dai un’occhiata alle chiavi
print("Chiavi nel pickle:", list(ds.keys()))

#%%
import numpy as np
import matplotlib.pyplot as plt

varname = "coh_uw"
co = ds["coherence"]
t0 = co.time.values[9]
h0 = co.heights.values[0]
da = co[varname].sel(time=t0, heights=h0)   # (freq,)

f = co.freq.values
y = da.values

# bande logaritmiche (40 bande)
fmin = max(1e-3, f[f > 0].min())
fmax = f.max()
bins = np.logspace(np.log10(fmin), np.log10(fmax), 41)

# median per bin
binned = da.groupby_bins(co.freq, bins=bins).median(dim="freq")

# prendi l'IntervalIndex e calcola i centri (media geometrica)
idx = binned["freq_bins"].to_index()  # pandas.IntervalIndex
left = idx.left.to_numpy()
right = idx.right.to_numpy()
centers = np.sqrt(left * right)

# alcuni bin possono essere vuoti -> NaN: filtra
mask = np.isfinite(binned.values)
centers = centers[mask]
binned_vals = binned.values[mask]

plt.figure(figsize=(9,6))
plt.semilogx(f, y, alpha=0.3, label=f"{varname} raw")
plt.semilogx(centers, binned_vals, linewidth=2, label=f"{varname} (log-bin median)")

# evidenzia banda microbarom
plt.axvspan(0.1, 0.6, alpha=0.2, label="microbarom band")

plt.ylim(0, 1.05)
plt.xlabel("Frequenza [Hz]")
plt.ylabel("Coerenza [-]")
plt.title(f"Coerenza Welch – time={str(t0)} – height={h0}")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()

