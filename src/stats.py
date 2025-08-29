#import standar libaries
import warnings
import numpy as np

#import third part libraries
import xarray as xr
from tqdm import tqdm
from scipy import signal


#import local libraries


def detrend(ds, window):
    """
    Remove the linear trend from the dataset for each time window block.

    Parameters:
    -----------
    ds : xarray.Dataset
        The input dataset.
    window : str
        Time window (e.g., "15min") for resampling.

    Returns:
    --------
    detrended_ds : xarray.Dataset
        A new dataset in which each block (resampled by the given window) has been detrended
        using linear detrending.
    """
    ds_groups = ds.resample(time=window)
    group_list = []

    for label, group in ds_groups:
        group_list.append(group.map(signal.detrend, args=[0]))

    return xr.concat(group_list, dim='time')



# ----------------------------------
def fill_gaps(ds, config, count_nans=True):
    """
    Fill gaps in an xarray Dataset based on the specified gap filling method.
    
    This function processes the dataset by resampling over a specified time window
    (provided in the configuration) and applies a gap filling strategy defined by the
    'gap_filling' key in the configuration dictionary. 
    
    The gap filling procedure for each resampled time segment includes:
      1. Counting the number of missing values (non-finite values) for each variable.
      2. Replacing infinities with NaN.
      3. Performing an interpolation along the "time" dimension with a limit of 10 consecutive NaNs.
      4. For any variable still containing NaNs after interpolation, replacing these with the segment 
         mean, and if NaNs persist (e.g., when the entire segment is NaN), replacing them with zero.
      5. Issuing a warning (only once per time segment) if there are still NaN values after interpolation.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        The input dataset containing one or more data variables that might have missing values.
    config : dict
        A configuration dictionary with at least the following keys:
          - "window": A string representing the resampling window (e.g., "15min").
          - "gap_filling": A string specifying the gap filling method ("test or "interp").
    count_nans : bool, optional (default=True)
        If True, the function also computes and returns a DataArray indicating the fraction of NaN
        values (QC) in each resampled time window.
    
    Returns:
    --------
    ds : xarray.Dataset
        The gap-filled dataset after processing all segments.
    nan_perc : xarray.DataArray
        A DataArray showing the fraction of NaN values per time window (QCnan). Returned only if 
        count_nans is True.
    
    Notes:
    ------
    - If the method specified in config["gap_filling"] is "test, the function calls
      `fill_gaps_test` to process the dataset.
    - If an unrecognized method is provided, the function issues a warning and defaults to 
      the "interp" method.
    - The function iterates over each time window segment using a progress bar provided by tqdm.
    - For each segment, missing values are first identified, then filled by interpolation, then
      any remaining missing entries are replaced with the segment mean and, if necessary, with zero.
    
    Raises:
    -------
    ValueError
        If a required variable does not meet the expected criteria during processing.
        
    Author: Samuele Mosso (with efforts from Mauro Ghirardelli)
    Last Change: 15/05/25
    """
    # config
    window = config["window"]
    method = config["gap_filling"]

    # check method
    if method != "interp":
        warnings.warn(
            "Gap filling method {} not recognized, will use interpolation".format(
                method
            )
        )
        method = "interp"
        
    # count nans per period
    var_list = list(ds.data_vars)
    ds_groups = ds.resample(time=window)
    nans = []
    gap_fill = []
    nan_warned = False

    # main loop
    for label, group in ds_groups:
        # count nans
        nan_array = np.invert(np.isfinite(group[var_list[0]]))
        for var in var_list[1:]:
            nan_array = nan_array | np.invert(np.isfinite(group[var]))
        nans.append(nan_array)

        # FILL GAPS
        # remove infs
        for var in var_list:
            group[var] = group[var].where(np.isfinite(group[var]), other=np.nan)

        # gap filling
        if method == "interp":
            group = group.interpolate_na(dim="time", limit=10)

        # take care of residual nans
        for var in var_list:
            if (np.isnan(group[var])).sum() > 0:
                # put to mean what is still nan after interpolate
                group[var] = group[var].where(
                    np.isfinite(group[var]), other=group[var].mean(axis=0)
                )
                # if nans are surviving it means the period was empty, put surviving nans to zero
                group[var] = group[var].where(np.isfinite(group[var]), other=0)
                # surviving nans warning, only warn once per period
                if not nan_warned:
                    warnings.warn(
                        "Nans are surviving gap filling in time period {}, set to mean value. Will only warn once.".format(
                            group.time[0].data
                        )
                    )
                    nan_warned = True
        gap_fill.append(group)

    # reconcatenate
    ds = xr.concat(gap_fill, dim="time")
    nans = xr.concat(nans, dim="time").resample(time=window)
    nan_perc = (nans.sum() / nans.count()).rename("QCnan")

    if count_nans:
        return ds, nan_perc
    else:
        return ds





def fill_gaps_test(ds, config, count_nans=True):
    """
    Fill gaps in an xarray Dataset using the "test" gap filling approach.
    
    This function processes the dataset in time windows (e.g., "15min") and applies the following steps:
    
    1. Replace infinities with NaN for each variable.
    2. Interpolate NaN values along the "time" dimension (limit 10 consecutive NaNs). 
       (If the entire window is NaN, no interpolation is performed.)
    3. Compute the percentage of missing values (QC) for all the variables
       for each segment. (A warning could be issued if the maximum QC exceeds 50%, but the segment is not discarded.)
    4. For each variable in every segment (separately), replace any remaining NaNs with the segment mean.
       If NaNs still exist after that (for example, if the segment is entirely NaN), then replace them with 0.
    5. Concatenate all time windows.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset containing data variables with potential gaps.
    config : dict
        Configuration dictionary containing:
            - "window": A string defining the resampling window (e.g., "15min").
            - "gap_filling": Should be set to "test" to trigger this specific approach.
    count_nans : bool, optional (default=True)
        If True, returns a DataArray with the fraction of NaN values (QCnan) in each time window.
    
    Returns:
    --------
    ds_filled : xarray.Dataset
        The dataset after gap filling over all segments.
    nan_perc : xarray.DataArray, optional
        A DataArray showing the fraction of NaN values in each window (if count_nans is True).
    
    Notes:
    ------
    - Instead of discarding segments with >50% missing data in any one of the required variables, 
      the function fills gaps by replacing NaNs first with the segment mean and then with 0.
    
    ----- Author: Mauro Ghirardelli (adapted from original effort by Samuele Mosso)
    """
    window = config["window"]
    var_list = list(ds.data_vars)
    

    # Resample the dataset by the specified time window.
    ds_groups = ds.resample(time=window)
    nans_masks = []  # To store NaN masks for QC calculation.
    processed_segments = []  # To store processed segments.

    # Loop over each time window (segment) using a progress bar.
    for label, group in tqdm(ds_groups, desc="Gapfilling segments", unit="seg"):
        # --- STEP 1: Replace infinities with NaN for each variable ---
        for var in var_list:
            group[var] = group[var].where(np.isfinite(group[var]), other=np.nan)

        # --- STEP 2: Interpolate NaN values (limit to 10 consecutive NaNs) ---
        group = group.interpolate_na(dim="time", limit=10)
        
        # --- QC Calculation for Required Variables (for informational purpose) ---
        qc_dict = {}
        for var in var_list:
            na_count = group[var].isnull().sum().item()
            total = group[var].size
            qc_dict[var] = na_count / total if total > 0 else 1.0
        max_qc = max(qc_dict.values())
        if max_qc > 0.50:
            warnings.warn(f"Segment {label}: high missing data detected, QC = {qc_dict}. Filling with mean/0.")

        # --- (Optional) Save the NaN mask for QC evaluation ---
        nan_mask = group[var_list[0]].isnull()
        for var in var_list[1:]:
            nan_mask = nan_mask | group[var].isnull()
        nans_masks.append(nan_mask)

        # --- STEP 3: Post-Interpolation Gap Handling ---
        for var in var_list:
            # Replace remaining NaNs with the segment mean.
            mean_val = group[var].mean(skipna=True)
            group[var] = group[var].fillna(mean_val)
            # Replace any remaining NaNs (if the segment is entirely NaN) with 0.
            group[var] = group[var].fillna(0)
        
        processed_segments.append(group)

    if not processed_segments:
        raise ValueError("No segments were processed.")
    
    
    # Concatenate all processed segments along the "time" dimension.
    ds_filled = xr.concat(processed_segments, dim="time")
    
    # Concatenate NaN masks and calculate overall QCnan if needed.
    if nans_masks:
        nans_concat = xr.concat(nans_masks, dim="time").resample(time=window)
        nan_perc = (nans_concat.sum() / nans_concat.count()).rename("QCnan")
    else:
        nan_perc = None

    if count_nans:
        return ds_filled, nan_perc
    else:
        return ds_filled
    



