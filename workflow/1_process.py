

#import standar libaries
import warnings
import sys
import os

#import third part libraries
import xarray as xr
import json
import pandas as pd
import pickle

#import local libraries
sys.path.append('/Users/mauro_ghirardelli/Documents/windpy4/src/')


from stats import fill_gaps, fill_gaps_test
from rotations import double_rotation
from fluxes import fluxes_calc
from scales import stationarity
from spectral_analysis import spectra_eps, coherence_from_spectra
from autocorrelation import autocorrelation
from microbarom import get_microbarom
from ogive import compute_ogive, band_ogives_logspace

from anisotropy import anisotropy_barycentric_ds
"""
from metutils import N2


from MRD import multiresolution

"""

def process(ds, config):
    """Function to postprocess the data from sonic anemometers.
    Author Samuele Mosso (with efforts from Mauro Ghirdelli), University of Innsbruck, last change 15/05/25
    
    INPUT:
        ds: An xarray dataset containing the variables u, v, w, and tc, with dimensions 'time' and 'heights'.
            If only one height is present, provide a 'heights' dimension with length one, as the height is needed for
            spectral cutoff determination.
        config: A configuration dictionary with the following entries:
            'window': Window size for averaging, given as 'nmin' with n being an integer.
            'avg_method': 'detrend' or 'block' for linear detrending or block averaging.
            'gap_filling': Gap filling method; supported methods include 
                            - 'interp' ( standard interpolation) 
                            - 'ghirardelli' (a test method, aimed at trying out new gap-filling solutions)
            'spectra': Boolean flag to compute the spectra.
            'strfun': Boolean flag to compute the 2nd order structure functions.
            'autocorr': Boolean flag to compute the autocorrelation functions.
            'MRD': Boolean flag to compute the multiresolution flux decompositions.
    
    OUTPUT:
        A dictionary with the following keys:
            'stats': Dataset with statistics such as the Reynolds stress tensor, etc.
            'spectra': Dataset with the spectra.
            'strfun': Dataset with the second order structure functions.
            'autocorr': Dataset with the autocorrelation functions.
            'MRD': Dataset with the multiresolution flux decompositions.
    """


    # handle single sonic
    if "heights" not in ds.coords:
        ds = ds.assign_coords(heights=[1])
        warnings.warn(
            "Dimension heights was not present, added with fake value, this will cause an imprecise"
            " determination of the cutoff in spectral frequency"
        )

    results = {}
    method = config["gap_filling"]

    # ==========================
    # BLOCK 0: data prep 
    # ==========================
    # --- gapfilling 
    if method=='test':
        ds, nan_perc = fill_gaps_test(ds, config)
    else:
        ds, nan_perc = fill_gaps(ds, config)

    # --- rotations
    ds, rotation = double_rotation(ds, config)

    # ==========================
    # BLOCK 1: fluxes and stationarity calculation 
    # ==========================
    #---- fluxes
    fluxes = fluxes_calc(ds, config)

    #---- stationarity
    stat = stationarity(ds, config)

    #---- merge
    stats = xr.merge([fluxes, rotation, stat,  nan_perc])
    
    #statistics = xr.merge([stats, metutils])
    statistics = stats

    # ==========================
    # BLOCK 2: Spectra
    # ==========================
    # spectra
    if config["spectra"]:
        raw_spectra, welch_spectra, spectra, epsilon, slopes = spectra_eps(
            ds, config, fluxes.meanU,
            raw_segments=None, raw_overlap=0.0,
            welch_segments=8, welch_overlap=0.5
        )


        statistics = xr.merge([statistics, epsilon, slopes])
        
        #results["raw_spectra"] = raw_spectra          # FFT/periodogram "pure"
        #results["welch_spectra"] = welch_spectra      # Welch (3, 50%)
        results["spectra"] = spectra                  # SG sul RAW (smoothed)
        # NEW: classical coherence from RAW (pre-smoothing)
        results["coherence"]     = coherence_from_spectra(welch_spectra)

    
    
    if config["microbarom"]:
        mbf_ds = get_microbarom(spectra, stats)
        statistics['MB_fit'] = mbf_ds['area_fit']
        statistics['MB_peak'] = mbf_ds['area_peak']
        statistics['MB_peak_abs'] = mbf_ds['area_peak_abs']
    
    
    # autocorrelation
    if config["autocorr"]:
        autocorr, intlen = autocorrelation(ds, config, fluxes.meanU)
        statistics = xr.merge([statistics, intlen])
        results["autocorr"] = autocorr
    
    
    if config["anisotropy"]:
        #ogive_raw = compute_ogive(raw_spectra, n_points=6000)
        #ogive_ds2 = compute_ogive(results["binned_spectra"])
        ogive_smooth = compute_ogive(spectra, n_points=6000)
        ogive_band = band_ogives_logspace(spectra, n_points=6000)
        #results["ogive_raw"] = ogive_ds1
        #results["ogive_binned"] = ogive_ds2
        #results["ogive"] = ogive_ds3
        
        #results["anisotropy_raw"] = anisotropy_barycentric_ds(ogive_raw)
        results["anisotropy_smooth"] = anisotropy_barycentric_ds(ogive_smooth)
        results["anisotropy_band"] = anisotropy_barycentric_ds(ogive_band)

    """
    # structure functions
    if config["strfun"]:
        strfun, epsilon = structure_functions_epsilon(ds, config, fluxes.meanU)
        statistics = xr.merge([statistics, epsilon])
        results["strfun"] = strfun
    """
    # put to nan the zeros in empty data
    statistics = statistics.where(statistics.meanU > 0)
    
    # create results dictionary
    results["stats"] = statistics
    return results
  
    



station = 'st1'

# Load configuration file
config_path = '/Users/mauro_ghirardelli/Documents/windpy4/conf/config_10min.txt'
with open(config_path, 'r') as file:
    config = json.load(file)
    

file_path = f"{config['path']}{station}/hf_ds_full_{station}_biascorr.nc"


# Open the NetCDF file as an xarray Dataset
ds = xr.open_dataset(file_path)

# Group daily file using floor to round time coordinate
group_keys = ds['time'].dt.floor("D")

# Output folder
output_folder = os.path.dirname(file_path)

for day, ds_day in ds.groupby(group_keys):
    day_str = pd.to_datetime(day).strftime('%Y-%m-%d')
    output_path = os.path.join(output_folder, f"{day_str}_{config['window']}.pkl")

    # Skip processing if file already exists
    if os.path.exists(output_path):
        print(f"Skipping {day_str} (already processed)")
        continue

    print(f"\nProcessing day: {day_str}")
    ds_filled = process(ds_day, config)

    # Save the dictionary
    with open(output_path, 'wb') as f:
        pickle.dump(ds_filled, f)

    





