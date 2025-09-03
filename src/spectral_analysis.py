#import standar libaries
import numpy as np

#import third part libraries
import xarray as xr
from scipy import signal
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import scipy.stats as stats


#import local libraries
from fluxes import get_fluctuations


# Main routine
# -------------

def spectra_eps(ds, config, wspd, *, raw_segments=None, raw_overlap=0.0,
                welch_segments=3, welch_overlap=0.5):
    """
    Compute raw periodogram spectra, Welch spectra, smoothed spectra,
    and dissipation/slopes.

    Parameters
    ----------
    ds : xarray.Dataset
        Input high-frequency dataset.
    config : dict
        Configuration dictionary; must include key 'window'.
    wspd : float
        Mean wind speed (m/s).
    raw_segments : int or None, optional
        Number of Welch segments for raw spectra.
        If None, use a single segment (periodogram).
    raw_overlap : float, optional
        Overlap fraction for raw spectra (0.0 to 1.0).
    welch_segments : int, optional
        Number of Welch segments for welch_spectra (default 3).
    welch_overlap : float, optional
        Overlap fraction for welch_spectra (default 0.5).

    Returns
    -------
    raw_spectra : xarray.Dataset
        Periodogram or Welch spectra for raw settings.
    welch_spectra : xarray.Dataset
        Welch spectra with chosen segments and overlap.
    spectra : xarray.Dataset
        Savitzky–Golay smoothed version of raw spectra.
    epsilon : xarray.Dataset
        Dissipation rate estimates.
    slopes : xarray.Dataset
        Low- and high-frequency slopes.
    """
    window = config['window']
    ds_fluct = get_fluctuations(ds, config)

    # Options for raw and Welch spectra
    raw_opts = dict(segments=raw_segments, overlap=raw_overlap,
                    window='boxcar', detrend=False)
    welch_opts = dict(segments=welch_segments, overlap=welch_overlap,
                      window='hann', detrend='constant')

    # Compute spectra
    raw_spectra = spectra_raw(ds_fluct, window, spec_opts=raw_opts)
    welch_spectra = spectra_raw(ds_fluct, window, spec_opts=welch_opts)
    spectra = spectra_processed(raw_spectra)

    # Compute slopes and epsilon
    slopes, epsilon = spectral_slopes_epsilon(spectra, wspd)

    return raw_spectra, welch_spectra, spectra, epsilon, slopes





# Subroutines
# -------------
def spectra_raw(ds, window, spec_opts=None):
    """
    Calculates spectra and cospectra for a high-frequency (hf) dataset (xarray) by time window.
    The dataset must be already rotated and detrended. This function supports both single and 
    multiple sonic setups (with multiple heights) and includes an additional variable 'p'
    (e.g., pressure) if available.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing variables 'u', 'v', 'w', 'tc', and optionally 'p'. 
        It should have coordinate dimensions 'time' and 'heights' (if multiple sonics).
    window : str or pandas offset
        Window specification used for resampling the dataset in time.
    spec_opts : dict, optional
        Keyword arguments forwarded to `calc_spectrum` and `calc_cospectrum`
        (e.g., {'segments': 3, 'overlap': 0.5, 'window': 'hann', 'detrend': 'constant'}).
        If None, the default behavior matches a single-segment periodogram.

    Returns
    -------
    spectra : xarray.Dataset
        Dataset with coordinates 'time', 'freq', and 'heights' (if multiple sonics).
        Data variables include:
            - su, sv, sw, sT: spectra for u, v, w, and sonic temperature (tc)
            - stke: spectrum of turbulent kinetic energy proxy (0.5*(u^2+v^2+w^2))
            - sp: spectrum for p (if 'p' exists in the input dataset)
            - cuw, cuv, cvw, cwT, cvT, cuT: cospectra between variables (u, v, w, tc)
            - cup, cvp, cwp, cTp: cospectra between variables and p (if pressure available)

    Notes
    -----
    The function will have issues if the data length is not an exact multiple of the window size.
    """
    if spec_opts is None:
        spec_opts = {}

    # check if multiple heights are present
    if len(np.shape(ds.u)) == 1:
        single_sonic = True
    else:
        single_sonic = False

    # resample and loop
    ds_groups = ds.resample(time=window)
    spectra = []
    
    # # one sonic
    if single_sonic:
        for label, group in tqdm(ds_groups, desc="Spectra calculation"):
            u = group.u
            v = group.v
            w = group.w
            tc = group.tc
            tke = 0.5 * (u**2 + v**2 + w**2)
            p = group.p if 'p' in group else None
                
            #==========
            # spectra
            #==========
            freq, su = calc_spectrum(u, **spec_opts)
            freq, sv = calc_spectrum(v, **spec_opts)
            freq, sw = calc_spectrum(w, **spec_opts)
            freq, sT = calc_spectrum(tc, **spec_opts)
            freq, stke = calc_spectrum(tke, **spec_opts)
                
            # Compute spectrum for 'p' if available
            if p is not None:
                freq, sp = calc_spectrum(p, **spec_opts)
            else:
                sp = None

            #==========
            # cospectra
            #==========
            freq, cuw = calc_cospectrum(u, w, **spec_opts)
            freq, cvw = calc_cospectrum(v, w, **spec_opts)
            freq, cuv = calc_cospectrum(u, v, **spec_opts)
            freq, cuT = calc_cospectrum(u, tc, **spec_opts)
            freq, cvT = calc_cospectrum(v, tc, **spec_opts)
            freq, cwT = calc_cospectrum(w, tc, **spec_opts)
            freq, ctkeT = calc_cospectrum(tke, tc, **spec_opts)
            
            if p is not None:
                freq, cup = calc_cospectrum(u, p, **spec_opts)
                freq, cvp = calc_cospectrum(v, p, **spec_opts)
                freq, cwp = calc_cospectrum(w, p, **spec_opts)
                freq, cTp = calc_cospectrum(tc, p, **spec_opts)
                freq, ctkep = calc_cospectrum(tke, p, **spec_opts)
            else:
                cup = None
                cvp = None
                cwp = None
                cTp = None
                ctkep = None

            # Assemble the computed spectra and cospectra into an xarray.Dataset.
            data_vars = dict(
                su=(['freq'], su),
                sv=(['freq'], sv),
                sw=(['freq'], sw),
                sT=(['freq'], sT),
                stke=(['freq'], stke),
                cuw=(['freq'], cuw),
                cuv=(['freq'], cuv),
                cvw=(['freq'], cvw),
                cwT=(['freq'], cwT),
                cvT=(['freq'], cvT),
                cuT=(['freq'], cuT),
                ctkeT=(['freq'], ctkeT),
            )
            if sp is not None:
                data_vars['sp'] = (['freq'], sp)
                data_vars['cup'] = (['freq'], cup)
                data_vars['cvp'] = (['freq'], cvp)
                data_vars['cwp'] = (['freq'], cwp)
                data_vars['cTp'] = (['freq'], cTp)
                data_vars['ctkep'] = (['freq'], ctkep)
                
            spectra.append(xr.Dataset(coords=dict(freq=freq, time=label),
                                      data_vars=data_vars))
        spectra = xr.concat(spectra, dim='time').assign_coords(heights=ds.heights)

    # multiple sonics
    else:
        for label, group in tqdm(ds_groups, desc="Spectra calculation"):
            single_spectra = []
            for h in group.heights:
                grouph = group.sel(heights=h)
                
                u = grouph.u
                v = grouph.v
                w = grouph.w
                tc = grouph.tc
                tke = 0.5 * (u**2 + v**2 + w**2)
                p = grouph.p if 'p' in grouph else None

                #==========
                # spectra
                #==========
                freq, su = calc_spectrum(u, **spec_opts)
                freq, sv = calc_spectrum(v, **spec_opts)
                freq, sw = calc_spectrum(w, **spec_opts)
                freq, sT = calc_spectrum(tc, **spec_opts)
                freq, stke = calc_spectrum(tke, **spec_opts)
                
                if p is not None:
                    freq, sp = calc_spectrum(p, **spec_opts)
                else:
                    sp = None
                
                #==========
                # cospectra
                #==========
                freq, cuw = calc_cospectrum(u, w, **spec_opts)
                freq, cvw = calc_cospectrum(v, w, **spec_opts)
                freq, cuv = calc_cospectrum(u, v, **spec_opts) 
                freq, cuT = calc_cospectrum(u, tc, **spec_opts) 
                freq, cvT = calc_cospectrum(v, tc, **spec_opts) 
                freq, cwT = calc_cospectrum(w, tc, **spec_opts) 
                freq, ctkeT = calc_cospectrum(tke, tc, **spec_opts)
                
                if p is not None:
                    freq, cup = calc_cospectrum(u, p, **spec_opts)
                    freq, cvp = calc_cospectrum(v, p, **spec_opts)
                    freq, cwp = calc_cospectrum(w, p, **spec_opts)
                    freq, cTp = calc_cospectrum(tc, p, **spec_opts)
                    freq, ctkep = calc_cospectrum(tke, p, **spec_opts)
                else:
                    cup = None
                    cvp = None
                    cwp = None
                    cTp = None
                    ctkep = None
                
                data_vars = dict(
                    su=(['freq'], su),
                    sv=(['freq'], sv),
                    sw=(['freq'], sw),
                    sT=(['freq'], sT),
                    cuw=(['freq'], cuw),
                    cuv=(['freq'], cuv),
                    cvw=(['freq'], cvw),
                    cwT=(['freq'], cwT),
                    cvT=(['freq'], cvT),
                    cuT=(['freq'], cuT),
                    ctkeT=(['freq'], ctkeT)
                )
                if sp is not None:
                    data_vars['sp'] = (['freq'], sp)
                    data_vars['cup'] = (['freq'], cup)
                    data_vars['cvp'] = (['freq'], cvp)
                    data_vars['cwp'] = (['freq'], cwp)
                    data_vars['cTp'] = (['freq'], cTp)
                    data_vars['ctkep'] = (['freq'], ctkep)
                    
                # put in dataset
                single_spectra.append(xr.Dataset(coords=dict(freq=freq, time=label, heights=h),
                                                 data_vars=data_vars))
            spectra.append(xr.concat(single_spectra, dim='heights'))

        spectra = xr.concat(spectra, dim='time')
    return spectra


def spectra_processed(ds,
                      window_pts=11,
                      polyorder=3,
                      num_log_pts=1000,
                      n_bins=3000,
                      apply_binning=False):
    """
    Applica Savitzky–Golay agli spettri e (opzionalmente) fa il binning log.
    Se apply_binning=False, restituisce gli spettri smussati sulla griglia freq originale.
    """
    freq = ds.freq.values
    has_heights = 'heights' in ds.dims

    # output freq: se non binniamo, restiamo sulla griglia originale
    if apply_binning:
        freq_out, _ = reduce_spectrum_logscale(freq, freq, n_bins=n_bins)
        dims = ('time', 'heights', 'freq') if has_heights else ('time', 'freq')
        n_freq_out = len(freq_out)
    else:
        freq_out = freq
        dims = ('time', 'heights', 'freq') if has_heights else ('time', 'freq')
        n_freq_out = len(freq_out)

    out_vars = {}
    shape = (ds.sizes['time'], ds.sizes.get('heights', 1), n_freq_out)

    for name, da in ds.data_vars.items():
        data = np.real(da.values)  # forza reale
        out = np.full(shape, np.nan, dtype=float)

        for ti in range(ds.sizes['time']):
            if has_heights:
                for hi in range(ds.sizes['heights']):
                    spec = data[ti, hi, :]
                    if np.sum(np.isfinite(spec)) < 2:
                        continue
                    segmented_flag = is_cospectrum(name)
                    _, smoothed = Savitzky_Golay_log(freq, spec,
                                                     window_pts=window_pts,
                                                     polyorder=polyorder,
                                                     num_log_pts=num_log_pts,
                                                     segmented=segmented_flag)
                    if apply_binning:
                        _, reduced = reduce_spectrum_logscale(freq, smoothed, n_bins=n_bins)
                        out[ti, hi, :] = reduced
                    else:
                        # nessun binning: allinea alla griglia originale
                        out[ti, hi, :] = smoothed
            else:
                spec = data[ti, :]
                if np.sum(np.isfinite(spec)) < 2:
                    continue
                segmented_flag = is_cospectrum(name)
                segmented_flag = is_cospectrum(name)
                _, smoothed = Savitzky_Golay_log(freq, spec,
                                                 window_pts=window_pts,
                                                 polyorder=polyorder,
                                                 num_log_pts=num_log_pts,
                                                 segmented=segmented_flag)
                if apply_binning:
                    _, reduced = reduce_spectrum_logscale(freq, smoothed, n_bins=n_bins)
                    out[ti, 0, :] = reduced
                else:
                    out[ti, 0, :] = smoothed

        arr = out if has_heights else out.squeeze(axis=1)
        out_vars[name] = (dims, arr)

    coords = {'time': ds.time, 'freq': freq_out}
    if has_heights:
        coords['heights'] = ds.heights

    return xr.Dataset(data_vars=out_vars, coords=coords)




def spectral_slopes_epsilon(spectra, wspd, h=None):
    # calculate cutoff based on height and speed
    if h is None:
        cutoff = wspd / (2 * np.pi * spectra.heights)
    else:
        cutoff = wspd / (2 * np.pi * h)

    # loose cospectra
    spectra = spectra[['su', 'sv', 'sw', 'sT']].map(np.real)

    # HIGH frequency spectra
    # isolate frequencies higher than cutoff, not all the way down because of aliasing
    spectra_high = spectra.where((spectra.freq > cutoff) & (spectra.freq < spectra.freq[-8]))
    # move left limit to the first maximum
    spectra_high = spectra_high.where(spectra_high.freq > spectra_high.idxmax(dim='freq'))

    # LOW frequency spectra
    # isolate frequencies lower than cutoff
    spectra_low = spectra.where(spectra.freq < cutoff)

    # EPSILON
    # kolmogorov constant
    cu = 18 / 55 * 1.5
    cvw = cu * 4 / 3
    cT = 0.8

    epsU = (2 * np.pi / wspd * (spectra_high.freq ** (5 / 3) * spectra_high.su / cu) ** (3 / 2)).median(
        dim='freq').rename('epsU')
    epsV = (2 * np.pi / wspd * (spectra_high.freq ** (5 / 3) * spectra_high.sv / cvw) ** (3 / 2)).median(
        dim='freq').rename('epsV')
    epsW = (2 * np.pi / wspd * (spectra_high.freq ** (5 / 3) * spectra_high.sw / cvw) ** (3 / 2)).median(
        dim='freq').rename('epsW')
    epsT = ((2 * np.pi / wspd) ** (2 / 3) * spectra_high.freq ** (5 / 3) * spectra_high.sT * epsU ** (
            1 / 3) / cT).median(dim='freq').rename('epsT')

    # SLOPES
    # switch to logarithmic space
    spectra_high = np.log10(spectra_high).assign_coords(freq=np.log10(spectra_high.freq))
    spectra_low = np.log10(spectra_low).assign_coords(freq=np.log10(spectra_low.freq))

    # spectral slopes
    slopes_h = spectra_high.polyfit(dim='freq', deg=1).sel(degree=1).drop_vars('degree').rename(
        dict(su_polyfit_coefficients='slopeHU',
             sv_polyfit_coefficients='slopeHV',
             sw_polyfit_coefficients='slopeHW',
             sT_polyfit_coefficients='slopeHT'))

    slopes_l = spectra_low.polyfit(dim='freq', deg=1).sel(degree=1).drop_vars('degree').rename(
        dict(su_polyfit_coefficients='slopeLU',
             sv_polyfit_coefficients='slopeLV',
             sw_polyfit_coefficients='slopeLW',
             sT_polyfit_coefficients='slopeLT'))

    return xr.merge([slopes_h, slopes_l]), xr.merge([epsU, epsV, epsW, epsT])


# Helpers
# -------------

def _resolve_welch_params(n, segments=None, overlap=0.0):
    """
    Derive (nperseg, noverlap) for Welch/csd given the desired number of segments
    and the overlap fraction. If `segments` is None, returns (n, 0), i.e. a single
    full-length segment (periodogram-like behavior).

    Parameters
    ----------
    n : int
        Total number of samples in the time series.
    segments : int or None
        Desired number of Welch segments. If None, use a single segment of length n.
    overlap : float
        Overlap fraction between segments in [0, 1). Typical values: 0.0, 0.5.

    Returns
    -------
    nperseg : int
        Samples per segment.
    noverlap : int
        Samples of overlap between segments.
    """
    if not segments:
        return n, 0
    ov = float(overlap)
    ov = max(0.0, min(0.95, ov))
    denom = (1.0 + (segments - 1) * (1.0 - ov))
    nperseg = max(8, int(np.floor(n / denom)))
    noverlap = int(np.floor(ov * nperseg))
    return nperseg, noverlap


def calc_spectrum(var, dt=None, *, segments=None, overlap=0.0,
                  window='boxcar', detrend=False, scaling='density'):
    """
    Compute the power spectral density (PSD) using Welch's method, or a single-
    segment periodogram if `segments=None`.

    Parameters
    ----------
    var : xarray.DataArray
        Input time series with a 'time' coordinate.
    dt : float, optional
        Sampling interval in seconds. If None, inferred from 'time'.
    segments : int or None, optional
        Number of Welch segments. None => use whole series (periodogram-like).
    overlap : float, optional
        Overlap fraction between segments (0..1). Ignored if segments=None.
    window : str or array_like, optional
        Window type passed to scipy.signal.welch (e.g., 'boxcar', 'hann').
    detrend : str or bool, optional
        Detrending option for welch (e.g., False, 'constant', 'linear').
    scaling : {'density','spectrum'}, optional
        Scaling of the result.

    Returns
    -------
    freq : ndarray
        Frequencies (Hz), with f[0] removed.
    spectrum : ndarray
        PSD values corresponding to `freq`.
    """
    n = len(var)
    if dt is None:
        dt = (var.time[1] - var.time[0]).item() / 1e9
    x = np.asarray(var.values)
    fs = 1.0 / dt

    nperseg, noverlap = _resolve_welch_params(n, segments=segments, overlap=overlap)
    freq, spectrum = signal.welch(
        x, fs=fs, window=window, detrend=detrend,
        nperseg=nperseg, noverlap=noverlap, scaling=scaling
    )
    return freq[1:], spectrum[1:].real


def calc_cospectrum(var1, var2, dt=None, *, segments=None, overlap=0.0,
                    window='boxcar', detrend=False, scaling='density'):
    """
    Compute the (complex) cross-spectrum using Welch's method, or a single-
    segment estimate if `segments=None`.

    Parameters
    ----------
    var1, var2 : xarray.DataArray
        Input time series sharing the same 'time' coordinate.
    dt : float, optional
        Sampling interval in seconds. If None, inferred from 'time' of var1.
    segments : int or None, optional
        Number of Welch segments. None => use whole series (periodogram-like).
    overlap : float, optional
        Overlap fraction between segments (0..1). Ignored if segments=None.
    window : str or array_like, optional
        Window type passed to scipy.signal.csd (e.g., 'boxcar', 'hann').
    detrend : str or bool, optional
        Detrending option for csd (e.g., False, 'constant', 'linear').
    scaling : {'density','spectrum'}, optional
        Scaling of the result.

    Returns
    -------
    freq : ndarray
        Frequencies (Hz), with f[0] removed.
    cospectrum : ndarray (complex)
        Cross-spectral density corresponding to `freq`.
    """
    n = len(var1)
    if dt is None:
        dt = (var1.time[1] - var1.time[0]).item() / 1e9

    x = np.asarray(var1.values)
    y = np.asarray(var2.values)
    fs = 1.0 / dt

    nperseg, noverlap = _resolve_welch_params(n, segments=segments, overlap=overlap)
    freq, cospectrum = signal.csd(
        x, y, fs=fs, window=window, detrend=detrend,
        nperseg=nperseg, noverlap=noverlap, scaling=scaling
    )
    return freq[1:], cospectrum[1:]


def is_cospectrum(name: str) -> bool:
    """
    Identify whether a variable name refers to a co-spectrum (cross-spectrum)
    or to an auto-spectrum.

    Parameters
    ----------
    name : str
        Variable name (e.g., 'su', 'cuw', 'ctkeT').

    Returns
    -------
    bool
        True if it looks like a cospectrum, False if auto-spectrum.
    """
    n = name.lower()
    autos = ("su", "sv", "sw", "st", "sp", "stke")  # note: 'sT' -> 'st' after lower()
    if n in autos:
        return False
    keys = ("cuv", "cuw", "cvw", "cup", "cvp", "cwp", "cut", "cvt", "cwt", "ctket", "ctkep", "ctp")
    return any(k in n for k in keys)





def Savitzky_Golay_log(freq, spectrum,
                        num_log_pts=1000,
                        window_pts=11,
                        polyorder=3,
                        segmented=False):
    """
    SG su griglia log10(f) che preserva l'area ∫C(f) df.
    Funziona con valori negativi (co-spettri). Restituisce sulla griglia originale.

    Parameters
    ----------
    freq : array-like, Hz (positivi, non necessariamente uniformi)
    spectrum : array-like (può essere complesso: si usa Re)
    num_log_pts : int, punti per la griglia uniforme in log10(f)
    window_pts : int, finestra SG (verrà resa dispari e <= num_log_pts del segmento)
    polyorder : int, ordine polinomiale SG
    segmented : bool, se True applica SG separatamente tra zero-crossing

    Returns
    -------
    freq_out : array (uguale a freq in input)
    spec_out : array smussata (float, con NaN dove input non valido)
    """
    f_in = np.asarray(freq)
    y_in = np.asarray(spectrum)
    # usa SOLO la parte reale e lavora in float
    if np.iscomplexobj(y_in):
        y_in = np.real(y_in)
    y_in = y_in.astype(float, copy=False)

    # maschera valida (non scartare negativi!)
    m = (f_in > 0) & np.isfinite(y_in)
    f = f_in[m]
    y = y_in[m]
    out_full = np.full_like(y_in, np.nan, dtype=float)

    if f.size < max(polyorder + 2, 3):
        out_full[m] = y
        return f_in, out_full

    logf = np.log10(f)
    logfu = np.linspace(logf.min(), logf.max(), int(num_log_pts))
    fu = 10.0 ** logfu

    # Interpolo y su griglia log
    yi = np.interp(logfu, logf, y)
    # Δf in frequenza lineare sulla griglia log (pesi corretti)
    dfu = np.gradient(fu)
    # contributi d'area
    Au = yi * dfu

    # helper: SG su un array (eventualmente di segmento), preservando dimensioni
    def _sg(x, win, deg):
        w = int(win)
        if w % 2 == 0:
            w += 1
        w = max(deg + 2, w)
        w = min(w, len(x) - (1 - len(x) % 2))  # non superare la lunghezza e mantieni dispari
        if w < deg + 2:
            return x  # troppi pochi punti -> lascia stare
        if w % 2 == 0:
            w = max(3, w - 1)
        return savgol_filter(x, window_length=w, polyorder=deg, mode='interp')

    if segmented:
        # trova cambi di segno su yi (non su Au: lo zero è in C(f))
        sgn = np.sign(yi)
        # indici dove cambia il segno (ignorando zeri esatti)
        zc = np.where(np.diff(np.signbit(yi)))[0]
        starts = np.r_[0, zc + 1]
        ends = np.r_[zc + 1, yi.size]

        Au_s = np.empty_like(Au)
        for a, b in zip(starts, ends):
            Au_s[a:b] = _sg(Au[a:b], window_pts, polyorder)
    else:
        Au_s = _sg(Au, window_pts, polyorder)

    # densità lisciata
    yi_s = Au_s / dfu

    # rinormalizza l'area totale (preserva ∑Au)
    A_orig = float(np.nansum(Au))
    A_smoo = float(np.nansum(Au_s))
    if np.isfinite(A_orig) and np.isfinite(A_smoo) and A_smoo != 0.0:
        yi_s *= (A_orig / A_smoo)

    # riporta su griglia originale
    y_back = np.interp(np.log10(f_in), logfu, yi_s)
    out_full[m] = y_back[m]
    return f_in, out_full



def reduce_spectrum_logscale(freq, spectrum, n_bins=2000):
    f = np.asarray(freq)
    S = np.asarray(spectrum)

    # Non scartare i valori negativi!
    mask = (f > 0) & np.isfinite(S)
    f_valid = f[mask]
    S_valid = S[mask]

    log_bins = np.linspace(np.log10(f_valid.min()), np.log10(f_valid.max()), n_bins)
    f_red = 10**log_bins

    if len(f_valid) < 2:
        return f_red, np.full_like(f_red, np.nan)

    interp_func = interp1d(f_valid, S_valid, kind='linear', fill_value='extrapolate')
    S_red = interp_func(f_red)

    return f_red, S_red




# ---- BLOCCO coerenza

def coherence_from_spectra(welch_spectra: xr.Dataset) -> xr.Dataset:
    """
    Build Welch coherence directly from autospectra and cospectra contained in
    `welch_spectra`:
        Coh_xy = |S_xy|^2 / (S_xx * S_yy)

    Expected variables (compute only those that exist):
      autos: su, sv, sw, sT, (sp optional)
      co:    cuw, cvw, cuv, cwT, cvT, cuT, (cup, cvp, cwp, cTp optional)

    Returns
    -------
    xr.Dataset with dims matching `welch_spectra` (time, freq[, heights]) and
    data_vars like: coh_uw, coh_vw, coh_uv, coh_wT, coh_vT, coh_uT, (coh_up, coh_vp, coh_wp, coh_Tp).
    Values are clipped to [0, 1].
    """
    S = welch_spectra

    # mapping: (auto_x, auto_y, co_xy, out_name)
    pairs = [
        ('su', 'sw', 'cuw', 'coh_uw'),
        ('sv', 'sw', 'cvw', 'coh_vw'),
        ('su', 'sv', 'cuv', 'coh_uv'),
        ('sw', 'sT', 'cwT', 'coh_wT'),
        ('sv', 'sT', 'cvT', 'coh_vT'),
        ('su', 'sT', 'cuT', 'coh_uT'),
        # pressure (optional)
        ('su', 'sp', 'cup', 'coh_up'),
        ('sv', 'sp', 'cvp', 'coh_vp'),
        ('sw', 'sp', 'cwp', 'coh_wp'),
        ('sT', 'sp', 'cTp', 'coh_Tp'),
    ]

    out = {}
    for sxx, syy, sxy, name in pairs:
        if (sxx in S.data_vars) and (syy in S.data_vars) and (sxy in S.data_vars):
            # |Sxy|^2 / (Sxx * Syy), safe & clipped
            num = np.abs(S[sxy])**2
            den = (np.real(S[sxx]) * np.real(S[syy]))
            coh = xr.where(den > 0, (num / den), np.nan)
            out[name] = coh.clip(0.0, 1.0)

    if not out:
        # nothing to compute
        return xr.Dataset(coords=S.coords)

    ds_coh = xr.Dataset(out, coords=S.coords)

    # attrs (optional)
    for v in ds_coh.data_vars:
        ds_coh[v].attrs.update({'long_name': f"Coherence {v[4:].replace('_','-')}",
                                'units': '—', 'method': 'Welch from spectra'})
    if 'freq' in ds_coh.coords:
        ds_coh.freq.attrs.update({'long_name': 'frequency', 'units': 'Hz'})
    return ds_coh








# --- Samu's binning
def logbin_spectrum(freq, spectrum, N_bins=9000, stat='mean'):
    """
    Computes log-spaced binned spectrum.

    Parameters
    ----------
    freq : array_like
        1D array of original frequency values.
    spectrum : array_like
        1D array of spectral values corresponding to freq.
    N_bins : int, optional
        Number of logarithmically spaced bins (default is 80).
    stat : {'mean', 'median'} or callable, optional
        Statistic to apply within each bin. Use 'mean', 'median', or a custom function.

    Returns
    -------
    freq_bin : ndarray
        Array of length N_bins containing the center frequencies of each log-spaced bin.
    spec_bin : ndarray
        Array of length N_bins containing the binned spectrum values.
    """
    if stat == 'mean':
        statistic = np.mean
    elif stat == 'median':
        statistic = np.median
    else:
        statistic = stat

    edges = np.logspace(np.log10(freq[0]), np.log10(freq[-1]), N_bins + 1)
    freq_bin = (edges[1:] + edges[:-1]) / 2
    spec_bin = stats.binned_statistic(freq, spectrum, statistic=statistic,
                                      bins=edges).statistic
    return freq_bin, spec_bin




def spectra_binning(ds,
                    N_bin=80):
    """
    Preprocess each real-valued spectral variable in `ds` along 'freq'.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with dims ('time', ['heights'], 'freq').
    N_bin : int
        Number of log-spaced bins (only for method='binning').
    window_pts : int
        Window size for smoothing (odd integer; used in both smoothing methods).
    method : {'binning', 'smoothing', 'Savitzky_Golay'}
        - 'binning': use logbin_spectrum(freq, spec, N_bins=N_bin)
        - 'smoothing': use smooth_spectrum(freq, spec, window_pts)
        - 'Savitzky_Golay': use Savitzky_Golay(freq, spec, ...)
    polyorder : int
        Polynomial order for Savitzky-Golay smoothing.
    num_log_pts : int
        Number of interpolation points in log10(f) for Savitzky-Golay.

    Returns
    -------
    xarray.Dataset
        New dataset with the same coords, except 'freq' replaced by:
        - binned-center frequencies (binning), or
        - original freq array (smoothing/Savitzky_Golay).
    """
    freq = ds.freq.values
    has_heights = 'heights' in ds.dims

    edges = np.logspace(np.log10(freq[0]),
                        np.log10(freq[-1]),
                        N_bin + 1)
    freq_out = (edges[:-1] + edges[1:]) / 2
    n_out = N_bin
    

    # define dims
    dims = ('time', 'heights', 'freq') if has_heights else ('time', 'freq')
    out_vars = {}

    for name, da in ds.data_vars.items():
        data = np.real(da.values)

        # allocate output array
        shape = (ds.sizes['time'], ds.sizes['heights'], n_out) if has_heights else (ds.sizes['time'], n_out)
        out = np.empty(shape, dtype=data.dtype)

        # loop over time and height
        for ti in range(ds.sizes['time']):
            if has_heights:
                for hi in range(ds.sizes['heights']):
                    spec = data[ti, hi, :]
                    _, vals = logbin_spectrum(freq, spec, N_bins=N_bin)
                    out[ti, hi, :] = vals
            else:
                spec = data[ti, :]
               
                _, vals = logbin_spectrum(freq, spec, N_bins=N_bin)
                out[ti, :] = vals

        # squeeze heights if not present
        arr = out if has_heights else out.squeeze(axis=1)
        out_vars[name] = (dims, arr)

    # rebuild dataset
    coords = {'time': ds.time, 'freq': freq_out}
    if has_heights:
        coords['heights'] = ds.heights

    return xr.Dataset(data_vars=out_vars, coords=coords)















