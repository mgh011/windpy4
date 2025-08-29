#import standar libaries
import numpy as np

#import third part libraries
import xarray as xr

#import local libraries
from fluxes import get_fluctuations, covariance


def stationarity(ds, config):
    """
    Calculates starionarity on sub-intervals of a sixth of the length of the original window
    """

    # config
    window = config["window"]

    # define window of sub intervals
    sub_window = window.split("m")[0] + "0S"

    # take mean out and divide in time blocks
    ds_fluct = get_fluctuations(ds, config)
    ds_groups = ds_fluct.resample(time=window)
    stat = []

    for label, group in ds_groups:
        # block covariance
        group_cov = covariance(group).assign_coords(time=label)
        # divide in sub-samples
        sub_groups = group.resample(time=sub_window)
        sub_group_cov = []
        # stats for each subsample
        for lab, sub_group in sub_groups:
            sub_group_cov.append(covariance(sub_group))
        # mean between subsamples
        sub_group_cov = (
            xr.concat(sub_group_cov, dim="time")
            .mean(dim="time")
            .assign_coords(time=label)
        )
        # stationarity tests
        stat.append(100 * np.abs((group_cov - sub_group_cov) / group_cov))

    return xr.concat(stat, dim="time").rename(dict(U="statU", uw="statUW", wT="statWT"))




