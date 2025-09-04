#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 16:14:46 2025

@author: mauro_ghirardelli
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# --- Example spectrum (synthetic) ---
f = np.linspace(0, 5, 500)
S = np.exp(-f) * np.sin(2*np.pi*f/5)**2 + 0.1*np.exp(-0.5*(f-2.5)**2/0.1)

# --- Prepare figure with two panels ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax1, ax2 = axes

ax1.plot(f, S, 'k', lw=1.5)
ax1.set_title("Standard ogive (area 0 → f)")
ax1.set_xlabel("Frequency f [Hz]")
ax1.set_ylabel("Spectrum S(f)")

ax2.plot(f, S, 'k', lw=1.5)
ax2.set_title("Reverse ogive (area f → ∞)")
ax2.set_xlabel("Frequency f [Hz]")
ax2.set_ylabel("Spectrum S(f)")

fill1 = ax1.fill_between([], [], [], color='tab:blue', alpha=0.4)
fill2 = ax2.fill_between([], [], [], color='tab:orange', alpha=0.4)

ax1.set_ylim(0, 1.1*S.max())
ax2.set_ylim(0, 1.1*S.max())

# --- Animation update ---
def update(frame):
    global fill1, fill2
    for coll in [fill1, fill2]:
        coll.remove()

    f_cut = f[frame]

    # Standard ogive: area 0 → f_cut
    mask1 = f <= f_cut
    fill1 = ax1.fill_between(f[mask1], S[mask1], color='tab:blue', alpha=0.4)

    # Reverse ogive: area f_cut → ∞
    mask2 = f >= f_cut
    fill2 = ax2.fill_between(f[mask2], S[mask2], color='tab:orange', alpha=0.4)

    return fill1, fill2

ani = FuncAnimation(fig, update, frames=len(f), interval=40, blit=False)

# --- Save as GIF ---
ani.save("ogive_vs_reverse.gif", writer=PillowWriter(fps=20))
plt.close(fig)
#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ogiva a banda (moving band-pass integration) visualizzata come area sotto lo spettro.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# --- Example spectrum (synthetic) ---
f = np.linspace(0, 5, 500)
S = np.exp(-f) * np.sin(2*np.pi*f/5)**2 + 0.1*np.exp(-0.5*(f-2.5)**2/0.1)

# --- Parameters for moving band ---
band_width = 0.5   # Hz, larghezza della banda
n_frames = len(f)

# --- Figure ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(f, S, 'k', lw=1.5)
ax.set_title("Band Ogive")
ax.set_xlabel("Frequency f [Hz]")
ax.set_ylabel("Spectrum S(f)")
ax.set_ylim(0, 1.1*S.max())

# init empty fill
fill = ax.fill_between([], [], [], color='tab:green', alpha=0.4)

# --- Animation update ---
def update(frame):
    global fill
    fill.remove()

    f_center = f[frame]
    f1, f2 = f_center, f_center + band_width

    mask = (f >= f1) & (f <= f2)
    fill = ax.fill_between(f[mask], S[mask], color='tab:green', alpha=0.4)

    ax.set_title(f"Band Ogive: [{f1:.2f}, {f2:.2f}] Hz")

    return fill,

ani = FuncAnimation(fig, update, frames=n_frames, interval=40, blit=False)

# --- Save as GIF ---
ani.save("ogiva_banda.gif", writer=PillowWriter(fps=20))
plt.close(fig)
