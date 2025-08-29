#import standar libaries
import numpy as np

#import third part libraries
import matplotlib.pyplot as plt
import pandas as pd


#import local libraries
import xarray as xr

# Main routine
# -------------

def get_microbarom(
    raw_spectra,
    stats,
    f_min=0.1,
    f_max=0.6,
    fit_deg=3,
    fit_range=(0.01, 5),
    plot=False
):
    """
    Apply log-log polynomial fitting outside the microbarom band and extract residual power.

    Parameters
    ----------
    raw_spectra : xr.Dataset
        Dataset with variables:
        - 'sp': (time, heights, freq) spectral density [Pa²/Hz]
        - 'freq': 1D frequency [Hz]
    stats : xr.Dataset
        Dataset with same dimensions (not used here, kept for interface compatibility).
    f_min : float
        Lower bound of the microbarom band [Hz].
    f_max : float
        Upper bound of the microbarom band [Hz].
    fit_deg : int
        Degree of polynomial used for log-log fitting.
    fit_range : tuple of float
        Frequency range used to fit the background [Hz].
    plot : bool
        If True, plot spectrum and fit for each (time, height).

    Returns
    -------
    xr.Dataset
        Dataset with dimensions (time, heights) and variables:
        - 'area_fit': Area under the fitted curve within the microbarom band
        - 'area_peak': Area above the fit (positive part only)
        - 'area_peak_abs': Total absolute residual from the fit in the band
    """
    times = raw_spectra.time
    heights = raw_spectra.heights

    area_fit = np.full((len(times), len(heights)), np.nan)
    area_peak = np.full_like(area_fit, np.nan)
    area_peak_abs = np.full_like(area_fit, np.nan)
    
    from tqdm import tqdm

    time_vals = list(raw_spectra.time)
    height_vals = list(raw_spectra.heights)


    for ti, t in enumerate(tqdm(time_vals, desc="Microbarom calculation")):
        for hi, h in enumerate(height_vals):
            spectrum = raw_spectra['sp'].isel(time=ti, heights=hi)
            afit, apeak, apeak_abs = MB_area(
                spectrum=spectrum,
                f_min=f_min,
                f_max=f_max,
                fit_deg=fit_deg,
                fit_range=fit_range,
                plot=plot,
                time_val=t,
                height_val=h
            )
            area_fit[ti, hi] = afit
            area_peak[ti, hi] = apeak
            area_peak_abs[ti, hi] = apeak_abs

    return xr.Dataset(
        data_vars=dict(
            area_fit=(("time", "heights"), area_fit),
            area_peak=(("time", "heights"), area_peak),
            area_peak_abs=(("time", "heights"), area_peak_abs),
        ),
        coords=dict(
            time=times,
            heights=heights
        )
    )

    
    
# Subroutines
# -------------

def MB_area(
    spectrum,
    f_min=0.1,
    f_max=0.6,
    fit_deg=3,
    fit_range=(0.01, 5),
    plot=False,
    time_val=None,
    height_val=None
):
    """
    Compute microbarom signal using log-log polynomial fitting outside the microbarom band.

    Parameters
    ----------
    spectrum : xr.DataArray
        Spectrum slice with frequency as dimension 'freq'.
    f_min : float
        Lower bound of microbarom band [Hz].
    f_max : float
        Upper bound of microbarom band [Hz].
    fit_deg : int
        Degree of polynomial for fitting.
    fit_range : tuple
        Frequency range where the fit is evaluated.
    plot : bool
        If True, produce diagnostic plot.
    time_val : datetime-like, optional
        Timestamp for plot title.
    height_val : float or str, optional
        Height for plot title.

    Returns
    -------
    area_fit : float
        Area under the fitted curve in microbarom band.
    area_peak : float
        Area of positive excess over fit in band.
    area_peak_abs : float
        Total absolute deviation from fit in band.
    """
    freq = spectrum.freq.values
    sp = spectrum.values

    # Clean data
    valid = (np.isfinite(freq) & np.isfinite(sp) & (sp > 0))
    freq_valid = freq[valid]
    sp_valid = sp[valid]

    if freq_valid.size < fit_deg + 1:
        return np.nan, np.nan, np.nan

    # Fit mask
    fit_mask = (
        (freq_valid >= fit_range[0]) &
        (freq_valid <= fit_range[1]) &
        ((freq_valid < f_min) | (freq_valid > f_max))
    )
    f_fit = freq_valid[fit_mask]
    sp_fit = sp_valid[fit_mask]

    if f_fit.size < fit_deg + 1:
        return np.nan, np.nan, np.nan

    # Fit in log-log space
    logf = np.log10(f_fit)
    logsp = np.log10(sp_fit)
    coeffs = np.polyfit(logf, logsp, deg=fit_deg)

    # Evaluate fit over entire range
    fit_eval_mask = (freq_valid >= fit_range[0]) & (freq_valid <= fit_range[1])
    freq_eval = freq_valid[fit_eval_mask]
    sp_eval = sp_valid[fit_eval_mask]
    sp_fit_all = 10 ** np.polyval(coeffs, np.log10(freq_eval))

    # Microbarom band
    band_mask = (freq_eval >= f_min) & (freq_eval <= f_max)
    f_band = freq_eval[band_mask]
    sp_band = sp_eval[band_mask]
    fit_band = sp_fit_all[band_mask]

    if f_band.size == 0:
        return np.nan, np.nan, np.nan

    # Area computations
    area_fit = np.trapz(fit_band, x=f_band)
    area_peak = np.trapz(np.maximum(sp_band - fit_band, 0), x=f_band)
    area_peak_abs = np.trapz(np.abs(sp_band - fit_band), x=f_band)

    if plot:
        title = f"Time: {pd.to_datetime(time_val).strftime('%Y-%m-%d %H:%M:%S')} | Height: {height_val} m"

        plt.figure(figsize=(8, 4))
        plt.loglog(freq_valid, sp_valid, label='sp', alpha=0.7)
        full_fit = 10 ** np.polyval(coeffs, np.log10(freq_valid))
        plt.loglog(freq_valid, full_fit, '--', label='Log-Log Fit', alpha=0.9)
        plt.axvspan(f_min, f_max, color='gray', alpha=0.3, label='Microbarom band')
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Spectral Power [Pa²/Hz]")
        plt.title(title)
        plt.grid(True, which='both', ls='--', lw=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return area_fit, area_peak, area_peak_abs








# ---- Old Code
# --- define microbarom_area_polyfit (as before) ---
def microbarom_area(
    spec_ds,
    stat_ds,
    f_min=0.1,
    f_max=0.6,
    poly_deg=3,
    plot=False,
    station_name='Station',
    output_filepath=None,
    fit_weights='inv_freq'  # Options: None, 'inv_freq', 'amp', 'log_amp', 'band_boost'
):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    if 'sp' not in spec_ds:
        return np.nan, np.nan, np.nan

    freq = spec_ds.freq.values
    sp = spec_ds['sp'].values  # raw pressure spectrum [Pa²/Hz]

    if freq.size <= 40:
        return np.nan, np.nan, np.nan

    # Trim first 40 frequency points
    freq_trim = freq[40:]
    sp_trim = sp[40:]

    # Define fit mask (exclude microbarom band)
    fit_mask = (freq_trim < f_min) | (freq_trim > f_max)
    f_fit = freq_trim[fit_mask]
    sp_fit = sp_trim[fit_mask]

    # Clean NaNs/infs
    valid = np.isfinite(f_fit) & np.isfinite(sp_fit)
    f_fit = f_fit[valid]
    sp_fit = sp_fit[valid]
    if f_fit.size < poly_deg + 1:
        return np.nan, np.nan, np.nan

    # --- Weighting strategy ---
    w = None
    eps = 1e-10
    if fit_weights == 'inv_freq':
        w = 1 / (f_fit + eps)
    elif fit_weights == 'amp':
        w = sp_fit
    elif fit_weights == 'log_amp':
        w = np.log10(sp_fit + eps)
    elif fit_weights == 'band_boost':
        w = np.ones_like(f_fit)
        band_mask_fit = (f_fit >= f_min) & (f_fit <= f_max)
        w[band_mask_fit] *= 5.0

    # --- Fit in log-log space ---
    logf = np.log10(f_fit)
    logy = np.log10(sp_fit)
    coeffs = np.polyfit(logf, logy, poly_deg, w=w if w is not None else None)

    # Evaluate fit over trimmed range
    logf_trim = np.log10(freq_trim)
    fit_trim = 10 ** np.polyval(coeffs, logf_trim)

    # Band of interest
    band_mask = (freq_trim >= f_min) & (freq_trim <= f_max)
    f_band = freq_trim[band_mask]
    sp_band = sp_trim[band_mask]
    fit_band = fit_trim[band_mask]

    # Areas
    area_fit = np.trapz(fit_band, x=f_band)
    area_peak = np.trapz(np.maximum(sp_band - fit_band, 0), x=f_band)
    area_peak_abs = np.trapz(np.abs(sp_band - fit_band), x=f_band)

    # --- Plotting ---
    if plot:
        try:
            time_val_raw = spec_ds.coords["time"].values.item()
            time_val = pd.to_datetime(time_val_raw).strftime("%Y-%m-%d %H-%M")
        except Exception:
            time_val = "unknown_time"

        try:
            height_val = spec_ds.coords["heights"].values.item()
        except Exception:
            height_val = "unknown_height"

        title_str = f"{station_name} | Height: {height_val} | Time: {time_val}"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: spectrum and fit
        ax1.loglog(freq_trim, sp_trim, label='PSD data', alpha=0.6)
        ax1.loglog(freq_trim, fit_trim, '--', label=f'Poly fit (deg={poly_deg})')
        ax1.axvspan(f_min, f_max, color='gray', alpha=0.3, label='Microbarom band')
        ax1.set(
            xlabel='Frequency [Hz]',
            ylabel='PSD [Pa²/Hz]',
            title=title_str
        )
        ax1.legend()
        ax1.grid(True, which='both', linestyle=':')

        # Right: residuals in band
        ax2.semilogx(f_band, sp_band - fit_band, 'x', label='Residual (data - fit)')
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.set(
            xlabel='Frequency [Hz]',
            ylabel='Residual [Pa²/Hz]',
            title='Microbarom Band Residuals'
        )
        ax2.legend()
        ax2.grid(True, which='both', linestyle=':')

        plt.tight_layout()

        if output_filepath is not None:
            fig.savefig(output_filepath, dpi=150)

        plt.show()

    return area_fit, area_peak, area_peak_abs











