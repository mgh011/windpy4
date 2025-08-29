#import standar libaries
import numpy as np

#import third part libraries
import xarray as xr


#import local libraries




def double_rotation(ds, config, return_rotation=True):
    """
    Apply a double rotation to the raw wind data in an xarray Dataset.
    
    This function performs a double rotation on the wind components ('u', 'v', 'w') in the 
    dataset. The double rotation aligns the coordinate system with the mean wind direction
    computed over a specified time window. It returns the rotated dataset along with the 
    corresponding rotation angles if requested.
    
    The rotation process includes:
      1. Resampling the dataset over a time window (provided in the configuration) to calculate 
         the mean wind components.
      2. Computing the rotation angles:
         - theta: the angle in the horizontal plane determined from the 'u' and 'v' components.
         - phi: the vertical inclination angle based on the 'w' component and the horizontal wind 
                speed.
         - A directional angle ('dir') is computed as (270 - theta*180/π) mod 360.
      3. Reindexing and forward filling the sine and cosine values for theta and phi along the 
         original time axis.
      4. Rotating the raw wind components using the computed sines and cosines.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset containing raw wind variables (typically 'u', 'v', 'w') among others.
    config : dict
        Configuration dictionary that must include:
            - "window": A string representing the resampling window (e.g., "15min"), which 
                        determines the temporal resolution for computing the mean values.
    return_rotation : bool, optional (default=True)
        If True, the function returns a tuple containing the rotated dataset and the rotation 
        angles (as an xarray.Dataset). If False, only the rotated dataset is returned.
    
    Returns:
    --------
    If return_rotation is True:
        ds_rotated : xarray.Dataset
            The dataset with the rotated wind components.
        rotation : xarray.Dataset
            A dataset containing the computed rotation angles:
                - 'dir': The adjusted direction in degrees.
                - 'theta': The horizontal rotation angle (in radians).
                - 'phi': The vertical rotation angle (in radians).
    
    If return_rotation is False:
        ds_rotated : xarray.Dataset
            The dataset with the rotated wind components.
    
    Notes:
    ------
    - The rotation is performed using the mean wind data computed by resampling the dataset according
      to the specified window.
    - The function uses forward-filling (ffill) to align the computed trigonometric values along the 
      complete time dimension.
    
    Author: Samuele Mosso (with efforts from Mauro Ghirardelli)
    Last Change: 15/05/25
    """

    # config
    window = config["window"]

    # means
    ds_mean = ds.resample(time=window).mean()

    # rotation angles
    theta = np.arctan2(ds_mean.v, ds_mean.u)
    phi = np.arctan2(ds_mean.w, np.sqrt(ds_mean.u**2 + ds_mean.v**2))
    rotation = xr.Dataset(
        data_vars=dict(dir=(270 - theta * 180 / np.pi) % 360, theta=theta, phi=phi)
    )

    # sines and cosines
    ct = np.cos(theta).reindex(time=ds.time).ffill(dim="time")
    st = np.sin(theta).reindex(time=ds.time).ffill(dim="time")
    cp = np.cos(phi).reindex(time=ds.time).ffill(dim="time")
    sp = np.sin(phi).reindex(time=ds.time).ffill(dim="time")

    # rotate
    u_rot = ct * cp * ds.u + st * cp * ds.v + sp * ds.w
    v_rot = -st * ds.u + ct * ds.v
    w_rot = -ct * sp * ds.u - st * sp * ds.v + cp * ds.w
    ds = ds.assign(
        u=u_rot,
        v=v_rot,
        w=w_rot,
    )

    if return_rotation:
        return ds, rotation
    else:
        return ds
    


