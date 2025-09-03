#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 09:40:38 2025
@author: mauro_ghirardelli
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---- Parametri del segnale ----
fs = 1000                 # frequenza di campionamento [Hz]
T  = 6.0                  # durata totale [s]
N  = int(fs*T)
t  = np.arange(N)/fs

# Rumore di base (leggermente colorato tramite IIR semplice)
rng = np.random.default_rng(42)
white = rng.standard_normal(N)
alpha = 0.95
noise = np.zeros(N)
for i in range(1, N):
    noise[i] = alpha*noise[i-1] + (1-alpha)*white[i]

# ---- Burst periodico modulato ----
t1, t2 = 2.0, 2.5                      # intervallo in cui è attivo
burst = (t >= t1) & (t <= t2)

fc = 40.0                               # frequenza portante [Hz]
fm = 3.0                                # frequenza di modulazione [Hz]
A  = 1.2                                # ampiezza della portante
m  = 0.8                                # indice di modulazione

periodic = A * (1 + m*np.sin(2*np.pi*fm*t)) * np.sin(2*np.pi*fc*t)
periodic[~burst] = 0.0

# Envelope lenta che modifica il rumore durante il burst
mod_envelope = np.ones_like(t)
mod_envelope[burst] = 1.0 + 0.5*np.sin(2*np.pi*fm*t[burst])

# Segnale finale
signal = mod_envelope * noise + periodic

# ---- Salva dati opzionale ----
df = pd.DataFrame({"t_s": t, "signal": signal})
df.to_csv("random_signal_with_periodic_burst.csv", index=False)

# ---- Plots a schermo ----

# Segnale completo
plt.figure(figsize=(9, 4))
plt.plot(t, signal, linewidth=1)
plt.axvspan(t1, t2, alpha=0.15, color='orange')
plt.title(f"Random signal with periodic signal burst ({t1}-{t2} s)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

"""
# Zoom
plt.figure(figsize=(11, 4))
plt.plot(t[burst], signal[burst], linewidth=1)
plt.title("Zoom on periodic, amplitude-modulated section")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
"""

# FFT
freqs = np.fft.rfftfreq(N, d=1/fs)
X = np.fft.rfft(signal * np.hanning(N))
PSD = (np.abs(X)**2) / (fs*N)

plt.figure(figsize=(9, 4))
plt.semilogy(freqs, PSD)
plt.title("Quick-look FFT magnitude (for reference)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power")
plt.xlim(0, 200)
plt.grid(True)

plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt

# ---------- Assumo che hai già t, signal, fs ----------
N = len(signal)
M = 8           # numero segmenti
overlap = 0.5   # 50%

# calcola lunghezza segmento e step
L_float = N / (1 + (M-1)*(1-overlap))
L = int(np.floor(L_float))
L -= (L % 2)  # pari
step = int(L * (1 - overlap))
starts = [i*step for i in range(M)]
if starts[-1] + L > N:
    step = (N - L) // (M - 1)
    starts = [i*step for i in range(M)]

# finestra Hann per Welch
window = np.hanning(L)
U = (window**2).sum()  # normalizzazione

freqs = np.fft.rfftfreq(L, d=1/fs)
psd_segments = []

for s in starts:
    seg = signal[s:s+L]
    X = np.fft.rfft(window * seg)
    psd = (np.abs(X)**2) / (U * fs)
    psd_segments.append(psd)

psd_segments = np.array(psd_segments)
psd_welch = psd_segments.mean(axis=0)

# FFT dell'intero segnale
freqs_full = np.fft.rfftfreq(N, d=1/fs)
X_full = np.fft.rfft(np.hanning(N) * signal)
PSD_full = (np.abs(X_full)**2) / (fs * (np.hanning(N)**2).sum())

# ---------- Plot ----------
plt.figure(figsize=(9, 5))
plt.semilogy(freqs_full, PSD_full, label="Full-signal FFT")
plt.semilogy(freqs, psd_welch, label="Welch (avg over 8 segments)")
plt.title("Full FFT vs Welch PSD")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [units^2/Hz]")
plt.xlim(0, fs/2)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# ========== Generazione segnale d'esempio (commenta se hai già t, signal, fs) ==========
fs = 500     # Hz
T  = 5.0      # s
N  = int(fs*T)
t  = np.arange(N)/fs

# rumore leggermente colorato
rng = np.random.default_rng(42)
white = rng.standard_normal(N)
alpha = 0.95
noise = np.zeros(N)
for i in range(1, N):
    noise[i] = alpha*noise[i-1] + (1-alpha)*white[i]

# burst periodico AM
t1, t2 = 2.0, 2.5
burst = (t >= t1) & (t <= t2)
fc, fm, A, m = 40.0, 3.0, 1.2, 0.8  # portante, modulazione, ampiezza, indice
periodic = A*(1 + m*np.sin(2*np.pi*fm*t))*np.sin(2*np.pi*fc*t)
periodic[~burst] = 0.0
env = np.ones_like(t)
env[burst] = 1.0 + 0.5*np.sin(2*np.pi*fm*t[burst])  # leggera variazione del rumore nel burst
signal = env*noise + periodic
# ========================================================================================

# ========== Helper: PSD con finestra Hann e scala "corretta" in units^2/Hz ==========
def psd_hann(x, fs):
    N = len(x)
    w = np.hanning(N)
    U = np.sum(w**2)  # potenza della finestra per la normalizzazione
    X = np.fft.rfft(w * x)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    PSD = (np.abs(X)**2) / (U * fs)
    return freqs, PSD

# ========== FFT “naturale” (intero segnale) ==========
freqs_full, PSD_full = psd_hann(signal, fs)

# ========== Parametri Welch: L = N/8, overlap 50% ==========
M_target = 8
L = N // 8
L -= (L % 2)             # pari
overlap = 0.5
step = int(L * (1 - overlap))  # con 50% -> L/2
if step < 1:
    step = 1

starts = []
s = 0
while s + L <= N:
    starts.append(s)
    s += step

# ========== Scegliamo due segmenti da visualizzare (uno all'inizio, uno sul burst) ==========
# primo segmento
idx_seg1 = 0

# segmento che contiene il centro del burst
t_mid = 0.5*(t1 + t2)
idx_mid = int(np.clip(int(t_mid*fs) - L//2, 0, N-L))
# trova l'indice in 'starts' più vicino a idx_mid
idx_seg2 = min(range(len(starts)), key=lambda k: abs(starts[k] - idx_mid))

# ========== Calcolo PSD per TUTTI i segmenti, e media Welch ==========
freqs_seg = np.fft.rfftfreq(L, d=1/fs)
psd_segments = []
for s0 in starts:
    seg = signal[s0:s0+L]
    _, psd = psd_hann(seg, fs)
    psd_segments.append(psd)
psd_segments = np.array(psd_segments)
PSD_welch = psd_segments.mean(axis=0)

# ========== PSD dei due segmenti scelti ==========
seg1 = signal[starts[idx_seg1]:starts[idx_seg1]+L]
seg2 = signal[starts[idx_seg2]:starts[idx_seg2]+L]
freqs_s, PSD_seg1 = psd_hann(seg1, fs)
_,       PSD_seg2 = psd_hann(seg2, fs)

# ========== PLOT 1: Segnale con burst e finestre dei due segmenti selezionati ==========
plt.figure(figsize=(9, 4))
plt.plot(t, signal, linewidth=1, label="signal")
#plt.axvspan(t1, t2, alpha=0.18, label="burst window")
# finestre segmenti
t_seg1_start = starts[idx_seg1]/fs
t_seg1_end   = (starts[idx_seg1]+L)/fs
t_seg2_start = starts[idx_seg2]/fs
t_seg2_end   = (starts[idx_seg2]+L)/fs

plt.title("Original signal")
plt.xlabel("Time [s]"); plt.ylabel("Amplitude")
plt.grid(True); plt.legend(loc="upper right")
plt.tight_layout()
#%%
plt.figure(figsize=(9, 4))
plt.plot(t, signal, linewidth=1, label="signal")
#plt.axvspan(t1, t2, alpha=0.18, label="burst window")
# finestre segmenti
t_seg1_start = starts[idx_seg1]/fs
t_seg1_end   = (starts[idx_seg1]+L)/fs
t_seg2_start = starts[idx_seg2]/fs
t_seg2_end   = (starts[idx_seg2]+L)/fs
plt.axvspan(t_seg1_start, t_seg1_end, alpha=0.15,color='red', label=f"segment 1 (L=N/8)")
plt.axvspan(t_seg2_start, t_seg2_end, alpha=0.15,color='green', label=f"segment 2 (around burst)")
plt.title("Original signal")
plt.xlabel("Time [s]"); plt.ylabel("Amplitude")
plt.grid(True); plt.legend(loc="upper right")
plt.tight_layout()

#%%
# ========== PLOT 2: FFT naturale (intero segnale) ==========
plt.figure(figsize=(11, 4))
plt.semilogy(freqs_full, PSD_full)
plt.title("Natural FFT (full-signal PSD)")
plt.xlabel("Frequency [Hz]"); plt.ylabel("PSD [units^2/Hz]")
plt.xlim(0, fs/2); plt.grid(True); plt.tight_layout()

#%%
# ========== PLOT 3: Le due FFT (PSD) dei segmenti selezionati ==========
plt.figure(figsize=(11, 4))
plt.semilogy(freqs_s, PSD_seg1,color='red', label="Segment 1 PSD")
plt.semilogy(freqs_s, PSD_seg2,color='green', label="Segment 2 PSD (near burst)")
plt.title("Two segment PSDs (L = N/8)")
plt.xlabel("Frequency [Hz]"); plt.ylabel("PSD [units^2/Hz]")
plt.xlim(0, fs/2); plt.grid(True); plt.legend(); plt.tight_layout()
#%%
# ========== PLOT 4: Welch finale (media di tutti i segmenti) ==========
plt.figure(figsize=(11, 4))
plt.semilogy(freqs_full, PSD_full, label="Natural FFT PSD")
plt.semilogy(freqs_seg, PSD_welch, label="Welch average PSD")
plt.title("Final Welch PSD (average over all segments, L = N/8, 50% overlap)")
plt.xlabel("Frequency [Hz]"); plt.ylabel("PSD [units^2/Hz]")
plt.xlim(0, fs/2); plt.grid(True); plt.legend(); plt.tight_layout()

plt.show()

# ---------- Stampa informazioni utili ----------
print(f"N={N}, fs={fs} Hz | segment length L={L}, step={step}, segments total={len(starts)}")
print(f"Shown segments: idx1={idx_seg1} (samples {starts[idx_seg1]}–{starts[idx_seg1]+L-1}), "
      f"idx2={idx_seg2} (samples {starts[idx_seg2]}–{starts[idx_seg2]+L-1})")



