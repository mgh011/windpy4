#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import standar libaries
import sys
import datetime
from tqdm import tqdm
import numpy as np

#import third part libraries
import xarray as xr
import pandas as pd

#import local libraries
sys.path.append('/Users/mauro_ghirardelli/Documents/windpy/src/')
from file_finder_reader import find_files_by_name_patterns, extract_date, read_TOB1, df_into_ds_consistent






#-------------------------------------
# READING/IMPORTING DATA FROM STATIONS
#-------------------------------------

"""
                    Slope top
+-----------------+-----------------+-----------------+
|  station_4      | station_5       | station_6       |  <-- top row
+-----------------+-----------------+-----------------+
|  station_1      | station_2       | station_3       |  <-- bottom row
+-----------------+-----------------+-----------------+
                    Slope bottom
                    

1) High frequency data (20Hz)):
    
        - sensor: CSAT3B sonic anemometer 
                  Nanobarometer
        - file format: TOB1
        -fields: "SECONDS",
                 "NANOSECONDS",
                 "RECORD",
                 "hf_pressure_lvl_1",     #Nanobarometer at 1m
                 "hf_pressure_lvl_2",     #Nanobarometer at 2m
                 "Ux","Uy","Uz","SonTemp","Diag", #Sonic at 1m
                 "Ux2","Uy2","Uz2","SonTemp2","Diag2" #Sonic at 2m
        - reader function: read_TOB1


2) Termocouples Data

3) MOMAA Station Data

"""

st1_data_path = "/Volumes/weop_hochhaeuser/station_2/LoggerNet/"   #station 5 data

# --------------
# 1) HF data
# --------------
st1_files = find_files_by_name_patterns(st1_data_path,"hf")
    
#st1_files = find_files_by_name_patterns(st1_data_path,"Digrtz")


# Define the start and end dates for filtering 
# Sensor status over the campaign: https://fileshare.uibk.ac.at/f/126e77fbd8864f5887f8/
start_date = datetime.datetime(2025, 1, 20)   # January 26, 2025
end_date   = datetime.datetime(2025, 2, 28)     # January 28, 2025

offset_combined= {
  'st1': {'SonTemp': 0.08, 'Ux': 0.01, 'Uy': -0.0, 'Uz': -0.01, 'U': 0.0, 'pressure': -0.19},
  'st2': {'SonTemp': 0.24, 'Ux': 0.04, 'Uy': 0.01, 'Uz': 0.01, 'U': 0.0, 'pressure': -0.195},
  'st3': {'SonTemp': -0.6, 'Ux': -0.06, 'Uy': -0.05, 'Uz': 0.01, 'U': 0.05, 'pressure': -0.27},
  'st4': {'SonTemp': -0.54, 'Ux': 0.02, 'Uy': 0.03, 'Uz': -0.02, 'U': -0.02, 'pressure': 0.07},
  'st5': {'SonTemp': 0.02, 'Ux': -0.03, 'Uy': 0.15, 'Uz': -0.02, 'U': -0.15, 'pressure': -0.14},
  'st6': {'SonTemp': -0.02, 'Ux': -0.01, 'Uy': -0.0, 'Uz': 0.01, 'U': -0.0, 'pressure': -0.17}
}


# Filter the files whose date is between start_date and end_date
selected_files = []
for filename in st1_files:
    file_date = extract_date(filename)
    if file_date and start_date <= file_date <= end_date:
        selected_files.append(filename)

# Print the selected files
#for f in selected_files:
#    print(f)

# Initialize an empty list to store the xarray datasets
ds_list = []

# Process each file with a progress bar
for path in tqdm(selected_files, desc="Concatenating raw files in one xarray ds"):
    df = read_TOB1(path)       # Function that reads the file and returns a DataFrame
    
    #ds = df_into_ds(df)      # Function that converts the DataFrame into an xarray Dataset with dimensions "time" and "height"
    ds = df_into_ds_consistent(df, invert_sonics=False)
    #ds  = df_into_ds_hf2(df)
    
    #ds =  df_into_ds_hf_seop(df)
    ds_list.append(ds)

# Concatenate the datasets along the time dimension.
# This works because all datasets share the same "height" dimension and have a time coordinate.
combined_ds = xr.concat(ds_list, dim="time")


#output_path = "/Users/mauro_ghirardelli/Documents/github/teamx_project/data/hf_ds.nc"

#combined_ds.to_netcdf(output_path)



#------------------------------------------------------------------------------
# READING HF NetCDF FILE
#------------------------------------------------------------------------------

# Path to the NetCDF file
#file_path = '/Users/mauro_ghirardelli/Documents/github/teamx_project/data/hf_ds.nc'

# Open the NetCDF file as an xarray Dataset.
#ds = xr.open_dataset(file_path)
ds = combined_ds
#------------------------------------------------------------------------------
# APPLY SATURATION MASK
#------------------------------------------------------------------------------

# Create a boolean mask for each cell (time, height)
# The mask is True if, for that timestamp and height, at least one of Ux, Uy, or Uz 
# has an absolute value greater than 65 m/s (the CSAT3B limit).
saturation_mask = ((np.abs(ds['Ux']) > 65) | (np.abs(ds['Uy']) > 65) | (np.abs(ds['Uz']) > 65))

# For each variable (Ux, Uy, Uz, SonTemp), replace values with NaN where the saturation mask is True.
for var in ['Ux', 'Uy', 'Uz', 'SonTemp']:
    ds[var] = ds[var].where(~saturation_mask)

#------------------------------------------------------------------------------
# APPLY OBSTRUCTION MASK
#------------------------------------------------------------------------------

# For variable "pressure", replace any value below 600 with NaN. 
# (600 is an arbitrary threshold chosen here.)
ds['pressure'] = ds['pressure'].where(ds['pressure'] >= 600)

#------------------------------------------------------------------------------
# APPLY FLAG MASK
#------------------------------------------------------------------------------

# Create a boolean mask for "Diag":
# For each (time, height) combination, if the value of Diag is 1, 2, or 4,
# then set the corresponding values in Ux, Uy, Uz, and SonTemp to NaN.
# Note: Please refer to the CSAT3B manual for details: 
# https://s.campbellsci.com/documents/ca/manuals/csat3b_man.pdf
diag_mask = ((ds['Diag'] == 1) | (ds['Diag'] == 2) | (ds['Diag'] == 4))

# Apply the diag_mask to Ux, Uy, Uz, and SonTemp.
for var in ['Ux', 'Uy', 'Uz', 'SonTemp']:
    ds[var] = ds[var].where(~diag_mask)

#------------------------------------------------------------------------------
# RENAME VARIABLES
#------------------------------------------------------------------------------

# Rename the variables for clarity and conciseness.
ds = ds.rename({
    'pressure': 'p',
    'Ux': 'u',
    'Uy': 'v',
    'Uz': 'w',
    'SonTemp': 'tc',
    'height': 'heights'
})

#------------------------------------------------------------------------------
# SORT DATASET BY TIME AND REMOVE DUPLICATES
#------------------------------------------------------------------------------

# Sort the dataset along the 'time' coordinate.
ds = ds.sortby("time")

# Remove duplicated time stamps, keeping only the first occurrence.
mask = ~ds.time.to_series().duplicated(keep='first')
ds = ds.isel(time=mask)

# Remove the "Diag" variable since it is no longer needed.
ds = ds.drop_vars("Diag")

# Print information about the dataset.
#print(ds)

#------------------------------------------------------------------------------
# REINDEX THE DATASET WITH A FULL TIME RANGE
#------------------------------------------------------------------------------

# Obtain the first and last timestamps from the dataset.
first_timestamp = ds['time'].values[0]
last_timestamp  = ds['time'].values[-1]

# Extract the dates (set time to midnight) from the first and last timestamps.
start_date = pd.to_datetime(first_timestamp).normalize()  # Yields "YYYY-MM-DD 00:00:00"
end_date   = pd.to_datetime(last_timestamp).normalize()    # Yields "YYYY-MM-DD 00:00:00" for the last day

# Define the start time as midnight of the start_date.
start_time = pd.to_datetime(f"{start_date} 00:00:00")

# Define the end time as "23:59:59.95" on the end_date.
# Note: At 20Hz (one sample every 50 ms), the last timestamp in a full day is 23:59:59.95.
end_time = pd.to_datetime(f"{end_date} 23:59:59.95")

# Create the complete time range at 20 Hz (i.e., with a frequency of 50 ms).
full_time = pd.date_range(start=start_time, end=end_time, freq="50ms")

# Make sure the dataset's 'time' coordinate is in datetime format.
ds['time'] = pd.to_datetime(ds['time'].values)

# Reindex the dataset using the new, complete time range.
# This will insert NaN for any missing timestamps.
full_ds = ds.reindex(time=full_time)

# Print the reindexed dataset information.
print(full_ds)


# Define cutoffdate
cutoff = pd.to_datetime("2025-02-14 14:00:00")

# build a cutoff mask 
time_mask = full_ds['time'] <= cutoff

# apply truncation to decimal numbers (multiplico, floor, poi divido)
truncated_p = np.floor(full_ds['p'] * 1e5) / 1e5

# replace values till cutoff
full_ds['p'] = full_ds['p'].where(~time_mask, truncated_p)



#------------------------------------------------------------------------------
# PLOT AN EXAMPLE VARIABLE (pressure 'p' for a given height)
#------------------------------------------------------------------------------

# Plot the variable 'p' at the height index 1.
#ds_full.p.isel(heights=1).plot()
#plt.title("Pressure (p) at Height Index 1")
#plt.xlabel("Time")
#plt.ylabel("Pressure")
#plt.show()

output_path = "/Users/mauro_ghirardelli/Documents/hf_ds_full_st2_weop.nc"

full_ds.to_netcdf(output_path)
#%%
import matplotlib.pyplot as plt
full_ds.u.isel(heights=1).plot()
plt.title("Pressure (p) at Height Index 1")
plt.xlabel("Time")
plt.ylabel("Pressure")
plt.show()




