#import standar libaries
import warnings
import numpy as np

#import third part libraries
import xarray as xr


#import local libraries
from stats import detrend


def get_fluctuations(ds, config):
    """
    Compute the fluctuations for each field in the dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        The input dataset containing fields for which fluctuations will be computed.
    config : dict
        Configuration dictionary that must contain:
          - 'avg_method': The method to compute fluctuations. Options are:
              'block'  -> Block averaging method.
              'detrend'-> Linear detrending using SciPy's detrend.
          - 'window': The time window (e.g., "15min") used for calculating block averages or detrending segments.
    
    Returns:
    --------
    ds_fluct : xarray.Dataset
        A new dataset with the fluctuations computed for each field and the original 'heights' coordinate preserved.

    Notes:
    ------
    For the 'block' method, the dataset is averaged over non-overlapping time windows, and the corresponding 
    fluctuations are computed as the difference between the raw values and the block average.
    
    For the 'detrend' method, each block of data is detrended using linear detrending (using scipy.signal.detrend), 
    which removes linear trends from the data.
    """
    # detrends each block and returns fluctuations
    method = config['avg_method']
    window = config['window']

    # avoid loss of heights dimension for single sonic
    heights = ds.heights

    # Compute fluctuations according to method
    if method == 'block':
        ds_mean = ds.resample(time=window).mean().reindex(time=ds.time).ffill(dim='time')
        ds_fluct = ds - ds_mean
    elif method == 'detrend':
        ds_fluct = detrend(ds, window)
    else:
        warnings.warn('avg_method {} not recognized, detrend will be used'.format(method))
        ds_fluct = detrend(ds, window)

    return ds_fluct.assign_coords(heights=heights)


def fluxes_calc(ds, config, third="all", fourth="main"):
    """
    Calculates statistics (turbulent fluxes) of specified orders at the chosen time average.
    
    This function computes various statistics and moment products of the fluctuating wind and temperature 
    fields. The method of computing the fluctuations can be selected via the configuration (e.g., by passing
    'block' or 'detrend' to the get_fluctuations function).
    
    The parameters 'third' and 'fourth' determine the orders of the statistics:
      - For third order:
          "all": compute all possible third order combinations (e.g., uuu, vvv, www, TTT, uuv, etc.).
          "main": compute only the main third order combinations (e.g., uuu, vvv, www, TTT).
          False: do not compute third order statistics.
      - For fourth order:
          If fourth is not False, fourth order statistics (e.g., uuuu, vvvv, wwww, TTTT) are computed.
          Note: Fourth order "all" is not implemented.
    
    **Additional functionality:**
    - If the dataset contains the key `"p"`, then the function will execute an additional routine,
      `flux_ghirardelli(ds, config)`, to compute supplementary flux estimates based on the pressure variable.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        The input dataset containing the variables used for flux calculations (typically 'u', 'v', 'w', 'tc',
        and optionally 'p').
    config : dict
        A configuration dictionary that must include at least:
            - "window": A string representing the averaging time window (e.g., "15min").
    third : str or bool, optional (default="all")
        Defines which third order products to compute:
            - "all": compute all possible third order combinations.
            - "main": compute only the main third order products (e.g., uuu, vvv, www, TTT).
            - False: third order statistics will not be computed.
    fourth : str or bool, optional (default="main")
        Defines which fourth order products to compute. If fourth is not False, the main fourth order products 
        (e.g., uuuu, vvvv, wwww, TTTT) are computed. (Note: fourth order "all" is not implemented.)
    
    Returns:
    --------
    fluxes : xarray.Dataset
        The dataset containing:
            - Second order fluxes and means:
                * meanU, meanT, uu, vv, ww, uv, uw, vw, TT, uT, vT, wT, sdir, tke.
            - Third order statistics (if third is not False):
                * uuu, vvv, www, TTT.
              And if third is "all", additional third order products such as uuv, uuw, uvw, etc.
            - Fourth order statistics (if fourth is not False):
                * uuuu, vvvv, wwww, TTTT.
            - ustar: computed as (uw**2 + vw**2)**0.25.
        Additionally, if the dataset contains the key "p", fluxes computed by the
        function `flux_ghirardelli(ds, config)` will be merged into the final dataset.
    
    Notes:
    ------
    - The function first computes the fluctuations using get_fluctuations.
    - It then constructs second order moments and calculates turbulent kinetic energy (TKE).
    - Based on the flags provided, it computes selected third and fourth order moments.
    - The computed products are averaged over the specified time window.
    - Finally, the friction velocity (ustar) is determined.
    
    Author: Samuele Mosso (with efforts from Mauro Ghirardelli)
    Last Change: 15/05/25
    """


    # Retrieve averaging window from configuration
    window = config["window"]

    # Compute fluctuations (e.g., via block or detrend methods)
    ds_fluct = get_fluctuations(ds, config)
    vars_present = set(ds.data_vars.keys())

    # Second order moments and means
    base = {
        "meanU":    ds.u,
        "meanT":    ds.tc,
        "uu":       ds_fluct.u * ds_fluct.u,
        "vv":       ds_fluct.v * ds_fluct.v,
        "ww":       ds_fluct.w * ds_fluct.w,
        "uv":       ds_fluct.u * ds_fluct.v,
        "uw":       ds_fluct.u * ds_fluct.w,
        "vw":       ds_fluct.v * ds_fluct.w,
        "TT":       ds_fluct.tc * ds_fluct.tc,
        "uT":       ds_fluct.u * ds_fluct.tc,
        "vT":       ds_fluct.v * ds_fluct.tc,
        "wT":       ds_fluct.w * ds_fluct.tc,
        "sdir":     np.arctan2(ds_fluct.v, ds_fluct.u)**2 * 180/np.pi,
        "tke":      0.5*(ds_fluct.u**2 + ds_fluct.v**2 + ds_fluct.w**2),
    }
    
    # if pressure exists, add those too
    if "p" in vars_present:
        base.update({
            "meanP": ds.p,
            "pp":  ds_fluct.p**2,
            "pu":  ds_fluct.p * ds_fluct.u,
            "pv":  ds_fluct.p * ds_fluct.v,
            "pw":  ds_fluct.p * ds_fluct.w,
            "pT":  ds_fluct.p * ds_fluct.tc,
        })
        
    fluxes = xr.Dataset(data_vars=base)


    # Third order statistics
    if third is not False:
        fluxes = fluxes.assign(
            uuu=ds_fluct.u * ds_fluct.u * ds_fluct.u,
            vvv=ds_fluct.v * ds_fluct.v * ds_fluct.v,
            www=ds_fluct.w * ds_fluct.w * ds_fluct.w,
            TTT=ds_fluct.tc * ds_fluct.tc * ds_fluct.tc,
        )
        # pressure cube, if available
        if "p" in vars_present:
            fluxes = fluxes.assign(
                ppp = ds_fluct.p   * ds_fluct.p   * ds_fluct.p,
            )

    if third == "all":
        fluxes = fluxes.assign(
            # — pure u/v/w/T cubes —
            uuu  = ds_fluct.u  * ds_fluct.u  * ds_fluct.u,
            uuv  = ds_fluct.u  * ds_fluct.u  * ds_fluct.v,
            uuw  = ds_fluct.u  * ds_fluct.u  * ds_fluct.w,
            uuT  = ds_fluct.u  * ds_fluct.u  * ds_fluct.tc,
    
            uvv  = ds_fluct.u  * ds_fluct.v  * ds_fluct.v,
            uvw  = ds_fluct.u  * ds_fluct.v  * ds_fluct.w,
            uww  = ds_fluct.u  * ds_fluct.w  * ds_fluct.w,
    
            vvv  = ds_fluct.v  * ds_fluct.v  * ds_fluct.v,
            vvw  = ds_fluct.v  * ds_fluct.v  * ds_fluct.w,
            vvT  = ds_fluct.v  * ds_fluct.v  * ds_fluct.tc,
    
            vww  = ds_fluct.v  * ds_fluct.w  * ds_fluct.w,
            www  = ds_fluct.w  * ds_fluct.w  * ds_fluct.w,
            wwT  = ds_fluct.w  * ds_fluct.w  * ds_fluct.tc,
    
            TTT  = ds_fluct.tc * ds_fluct.tc * ds_fluct.tc,
            uTT  = ds_fluct.u  * ds_fluct.tc * ds_fluct.tc,
            vTT  = ds_fluct.v  * ds_fluct.tc * ds_fluct.tc,
            wTT  = ds_fluct.w  * ds_fluct.tc * ds_fluct.tc,
    
            # — coupling with TKE —
            utke = fluxes.tke  * ds_fluct.u,
            vtke = fluxes.tke  * ds_fluct.v,
            wtke = fluxes.tke  * ds_fluct.w,
    
            # — mixed u-v-w-T triples —
            uvT  = ds_fluct.u  * ds_fluct.v  * ds_fluct.tc,
            uwT  = ds_fluct.u  * ds_fluct.w  * ds_fluct.tc,
            vwT  = ds_fluct.v  * ds_fluct.w  * ds_fluct.tc,
        )

        if "p" in vars_present:
            fluxes = fluxes.assign(
                # — two velocities + pressure —
                uup = ds_fluct.u * ds_fluct.u * ds_fluct.p,
                uvp = ds_fluct.u * ds_fluct.v * ds_fluct.p,
                uwp = ds_fluct.u * ds_fluct.w * ds_fluct.p,
                vvp = ds_fluct.v * ds_fluct.v * ds_fluct.p,
                vwp = ds_fluct.v * ds_fluct.w * ds_fluct.p,
                wwp = ds_fluct.w * ds_fluct.w * ds_fluct.p,
    
                # — one velocity + pressure + T —
                upT = ds_fluct.u * ds_fluct.p * ds_fluct.tc,
                vpT = ds_fluct.v * ds_fluct.p * ds_fluct.tc,
                wpT = ds_fluct.w * ds_fluct.p * ds_fluct.tc,
    
                # — pressure + two T’s —
                pTT = ds_fluct.p * ds_fluct.tc * ds_fluct.tc,
            )

       

    # Fourth order statistics
    if fourth is not False:
        fluxes = fluxes.assign(
            uuuu=ds_fluct.u * ds_fluct.u * ds_fluct.u * ds_fluct.u,
            vvvv=ds_fluct.v * ds_fluct.v * ds_fluct.v * ds_fluct.v,
            wwww=ds_fluct.w * ds_fluct.w * ds_fluct.w * ds_fluct.w,
            TTTT=ds_fluct.tc * ds_fluct.tc * ds_fluct.tc * ds_fluct.tc,
        )
        if "p" in vars_present:
             fluxes = fluxes.assign(
                 pppp=ds_fluct.p * ds_fluct.p * ds_fluct.p * ds_fluct.p
             )
    # Average over the specified time window
    fluxes = fluxes.resample(time=window).mean()

    # Compute friction velocity (ustar)
    fluxes = fluxes.assign(ustar=(fluxes.uw**2 + fluxes.vw**2) ** 0.25)


    return fluxes



def covariance(ds):
    """method used by Postprocess_whole.stationarity to calculate the subgroup covariances"""
    ds_cov = xr.Dataset(
        data_vars=dict(
            U=np.sqrt(
                (ds.u * ds.u).mean(dim="time")
                - ds.u.mean(dim="time") * ds.u.mean(dim="time")
            ),
            uw=(ds.u * ds.w).mean(dim="time")
            - ds.u.mean(dim="time") * ds.w.mean(dim="time"),
            wT=(ds.w * ds.tc).mean(dim="time")
            - ds.w.mean(dim="time") * ds.tc.mean(dim="time"),
        )
    )
    return ds_cov
