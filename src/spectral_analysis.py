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
def spectra_eps(ds, config, wspd):
    '''works on already rotated data
    At the moment the dissipation calculation for single sonic should work only if there is a height dimension with 1 value'''

    # config
    window = config['window']
    #processing_method = config['processing_method']
    

    # divide in time blocks
    ds_fluct = get_fluctuations(ds, config)

    # spectra & co
    raw_spectra = spectra_raw(ds_fluct, window)
    #binned_spectra = spectra_binning(raw_spectra)
    spectra = spectra_processed(raw_spectra)

    # slopes high and low freq and dissipation rates
    slopes, epsilon = spectral_slopes_epsilon(spectra, wspd)

    return raw_spectra, spectra, epsilon, slopes




# Subroutines
# -------------
def spectra_raw(ds, window):
    """
    Calculates spectra and cospectra for high-frequency (hf) dataset (xarray) by time window.
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
    
    Returns
    -------
    spectra : xarray.Dataset
        Dataset with coordinates 'time', 'freq', and 'heights' (if multiple sonics).
        Data variables include:
            - su, sv, sw, sT: spectra for u, v, w, and sonic temperature (tc)
            - sp: spectrum for p (if 'p' exists in the input dataset)
            - cuw, cuv, cvw, cwT, cvT, cuT: cospectra between variables (u, v, w, tc)
            - cup, cvp, cwp, cTp: cospectra between variables and p (if pressure data is available)
    
    Notes
    -----
    The function will have issues if the data length is not an exact multiple of the window size.
    """
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
            freq, su = calc_spectrum(u)
            freq, sv = calc_spectrum(v)
            freq, sw = calc_spectrum(w)
            freq, sT = calc_spectrum(tc)
            freq, stke = calc_spectrum(tke)
                
             # Compute spectrum for 'p' if available
            if p is not None:
                freq, sp = calc_spectrum(p)
            else:
                sp = None
            #==========
            # cospectra
            #==========
            freq, cuw = calc_cospectrum(u, w)
            freq, cvw = calc_cospectrum(v, w)
            freq, cuv = calc_cospectrum(u, v)
            freq, cuT = calc_cospectrum(u, tc)
            freq, cvT = calc_cospectrum(v, tc)
            freq, cwT = calc_cospectrum(w, tc)
            
            freq, ctkeT = calc_cospectrum(tke, tc)
            
            if p is not None:
                freq, cup = calc_cospectrum(u, p)
                freq, cvp = calc_cospectrum(v, p)
                freq, cwp = calc_cospectrum(w, p)
                freq, cTp = calc_cospectrum(tc, p)
                freq, ctkep = calc_cospectrum(tke, p)
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
                tke = 0.5 * (u**2 + v**2 + w**2)**0.5
                p = grouph.p if 'p' in grouph else None

                #==========
                # spectra
                #==========
                freq, su = calc_spectrum(u)
                freq, sv = calc_spectrum(v)
                freq, sw = calc_spectrum(w)
                freq, sT = calc_spectrum(tc)
                
                freq, stke = calc_spectrum(tke)
                
                if p is not None:
                    freq, sp = calc_spectrum(p)
                else:
                    sp = None
                
                #==========
                # cospectra
                #==========
                freq, cuw = calc_cospectrum(u,w)
                freq, cvw = calc_cospectrum(v,w)
                freq, cuv = calc_cospectrum(u,v) 
                freq, cuT = calc_cospectrum(u,tc) 
                freq, cvT = calc_cospectrum(v,tc) 
                freq, cwT = calc_cospectrum(w,tc) 
                
                freq, ctkeT = calc_cospectrum(tke,tc)
                
                if p is not None:
                    freq, cup = calc_cospectrum(u, p)
                    freq, cvp = calc_cospectrum(v, p)
                    freq, cwp = calc_cospectrum(w, p)
                    freq, cTp = calc_cospectrum(tc, p)
                    freq, ctkep = calc_cospectrum(tke, p)
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
def calc_spectrum(var, dt=None):
    #only real
    n = len(var)
    if dt is None:
        dt = (var.time[1] - var.time[0]).item() / 1e9
    freq, spectrum = signal.welch(var, fs=1 / dt,
                                  window='boxcar', detrend=False,
                                  nperseg=n, noverlap=0)
    freq = freq[1:]
    spectrum = spectrum[1:].real

    return freq, spectrum


def calc_cospectrum(var1, var2, dt=None):
    #imaginary number
    n = len(var1)
    if dt is None:
        dt = (var1.time[1] - var1.time[0]).item() / 1e9

    freq, cospectrum = signal.csd(var1, var2, fs=1 / dt,
                                  window='boxcar', detrend=False,
                                  nperseg=n, noverlap=0)
    freq = freq[1:]
    cospectrum = cospectrum[1:]

    return freq, cospectrum

def is_cospectrum(name: str) -> bool:
    n = name.lower()
    autos = ("su","sv","sw","sT","sp","stke")  # include your autos as needed (note: case-insensitive)
    if n in autos:
        return False
    # typical cross-spectra keys (cover uv, uw, vw, *p, *T, tke*)
    keys = ("cuv","cuw","cvw","cup","cvp","cwp","cuT","cvT","cwT","ctkeT","ctkep","cTp")
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

from scipy.signal import coherence
from fluxes import get_fluctuations

def _welch_params(N):
    nperseg = max(256, min(4096, N // 8))  # ~8 segmenti + 50% overlap
    noverlap = nperseg // 2
    return nperseg, noverlap

def _compute_welch_coh_pairs(data_dict, fs):
    keys = [k for k in ['u','v','w','tc','p'] if k in data_dict]
    m = np.ones_like(data_dict[keys[0]], dtype=bool)
    for k in keys:
        m &= np.isfinite(data_dict[k])
    for k in keys:
        data_dict[k] = data_dict[k][m]
    N = len(data_dict[keys[0]])
    if N < 256:
        return None

    nperseg, noverlap = _welch_params(N)
    out = {}
    f, c = coherence(data_dict['u'], data_dict['w'], fs=fs, window='hann',
                     nperseg=nperseg, noverlap=noverlap, detrend='constant')
    out['f'] = f; out['coh_uw'] = np.clip(c, 0.0, 1.0)

    _, c = coherence(data_dict['v'], data_dict['w'], fs=fs, window='hann',
                     nperseg=nperseg, noverlap=noverlap, detrend='constant')
    out['coh_vw'] = np.clip(c, 0.0, 1.0)

    _, c = coherence(data_dict['u'], data_dict['v'], fs=fs, window='hann',
                     nperseg=nperseg, noverlap=noverlap, detrend='constant')
    out['coh_uv'] = np.clip(c, 0.0, 1.0)

    _, c = coherence(data_dict['w'], data_dict['tc'], fs=fs, window='hann',
                     nperseg=nperseg, noverlap=noverlap, detrend='constant')
    out['coh_wT'] = np.clip(c, 0.0, 1.0)

    _, c = coherence(data_dict['v'], data_dict['tc'], fs=fs, window='hann',
                     nperseg=nperseg, noverlap=noverlap, detrend='constant')
    out['coh_vT'] = np.clip(c, 0.0, 1.0)

    _, c = coherence(data_dict['u'], data_dict['tc'], fs=fs, window='hann',
                     nperseg=nperseg, noverlap=noverlap, detrend='constant')
    out['coh_uT'] = np.clip(c, 0.0, 1.0)

    if 'p' in data_dict:
        _, c = coherence(data_dict['u'], data_dict['p'], fs=fs, window='hann',
                         nperseg=nperseg, noverlap=noverlap, detrend='constant')
        out['coh_up'] = np.clip(c, 0.0, 1.0)
        _, c = coherence(data_dict['v'], data_dict['p'], fs=fs, window='hann',
                         nperseg=nperseg, noverlap=noverlap, detrend='constant')
        out['coh_vp'] = np.clip(c, 0.0, 1.0)
        _, c = coherence(data_dict['w'], data_dict['p'], fs=fs, window='hann',
                         nperseg=nperseg, noverlap=noverlap, detrend='constant')
        out['coh_wp'] = np.clip(c, 0.0, 1.0)
        _, c = coherence(data_dict['tc'], data_dict['p'], fs=fs, window='hann',
                         nperseg=nperseg, noverlap=noverlap, detrend='constant')
        out['coh_Tp'] = np.clip(c, 0.0, 1.0)
    return out

def coherence_welch_from_ds(ds, config):
    """
    Calcola la coerenza (Welch) per finestre da config['window'] usando direttamente 'ds'
    (ruotato/detrendato), senza esporre ds_fluct al main.
    Ritorna un xr.Dataset con dims (time, freq[, heights]).
    """
    fs = 20
    window = config['window']
    single_sonic = (len(np.shape(ds.u)) == 1)

    # ottiene le fluttuazioni nello stesso modo del blocco spettri
    ds_fluct = get_fluctuations(ds, config)
    ds_groups = ds_fluct.resample(time=window)
    out_list = []

    if single_sonic:
        for label, g in ds_groups:
            if g.time.size < 256:
                continue
            d = {'u': g.u.values, 'v': g.v.values, 'w': g.w.values, 'tc': g.tc.values}
            if 'p' in g:
                d['p'] = g.p.values
            res = _compute_welch_coh_pairs(d, fs)
            if res is None:
                continue
            data_vars = {k: (['freq'], res[k]) for k in res if k != 'f'}
            out_list.append(xr.Dataset(coords=dict(freq=res['f'], time=label),
                                       data_vars=data_vars))
        if not out_list:
            return xr.Dataset()
        ds_coh = xr.concat(out_list, dim='time').assign_coords(heights=ds.heights)
    else:
        for label, g in ds_groups:
            per_h = []
            for h in g.heights:
                gh = g.sel(heights=h)
                if gh.time.size < 256:
                    continue
                d = {'u': gh.u.values, 'v': gh.v.values, 'w': gh.w.values, 'tc': gh.tc.values}
                if 'p' in gh:
                    d['p'] = gh.p.values
                res = _compute_welch_coh_pairs(d, fs)
                if res is None:
                    continue
                data_vars = {k: (['freq'], res[k]) for k in res if k != 'f'}
                per_h.append(xr.Dataset(coords=dict(freq=res['f'], time=label, heights=h),
                                        data_vars=data_vars))
            if per_h:
                out_list.append(xr.concat(per_h, dim='heights'))
        if not out_list:
            return xr.Dataset()
        ds_coh = xr.concat(out_list, dim='time')

    for v in ds_coh.data_vars:
        ds_coh[v].attrs.update({'long_name': f'Coherence {v[4:].replace("_","-")}',
                                'units': '—', 'method': 'Welch'})
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















