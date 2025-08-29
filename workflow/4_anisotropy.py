#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 10:02:30 2025

@author: mauro_ghirardelli
"""

#import standar libaries
import os
import pickle
import warnings
from pathlib import Path
import sys

#import third part libraries
import numpy as np
import matplotlib.pyplot as plt


#import local libraries
sys.path.append('/Users/mauro_ghirardelli/Documents/windpy3/src/')
from plotter import pdf_pp_f



# Cartella principale
base_dir = Path("/Users/mauro_ghirardelli/Documents/TEAMx/20250416_analisi/")

# Definizione dei gruppi di stazioni
group1 = {"st1", "st2", "st3"}
group2 = {"st4", "st5", "st6"}

# Liste finali
file_paths_group1 = []
file_paths_group2 = []

# Cerca tutti i .pkl nei sottodirectory
all_pickle_paths = base_dir.glob("**/*.pkl")

for path in all_pickle_paths:
    if "10min" not in path.stem.lower():
        continue  # salta se non contiene "hf"

    # Trova il nome della cartella stazione (es: "st2") cercando tra i genitori
    for parent in path.parents:
        if parent.name in group1:
            file_paths_group1.append(path)
            break
        elif parent.name in group2:
            file_paths_group2.append(path)
            break

# Controllo finale
print(f"📁 File in group1 (st1-3): {len(file_paths_group1)}")
print(f"📁 File in group2 (st4-6): {len(file_paths_group2)}")

#%%
import json
import xarray as xr
station = 'st6'

# Load configuration file
config_path = '/Users/mauro_ghirardelli/Documents/windpy3/conf/config_10min.txt'
with open(config_path, 'r') as file:
    config = json.load(file)
    

file_path = f"{config['path']}{station}/2025-02-05_10min.pkl"


with open(file_path, 'rb') as f:
    ds = pickle.load(f)

print(ds)
#%%
print(ds['ogive'])
ogive = ds['ogive']
#%%

# dopo aver ottenuto ogive:


# versione element-wise (niente loop espliciti):
from anisotropy import anisotropy_pointwise
ds_a = anisotropy_pointwise(ogive, return_rotation=True)
#%%
print(ds_a)
# --- Plot: Barycentric Triangle + Return-to-Isotropy curve --------------------
import numpy as np
import matplotlib.pyplot as plt

def plot_return_to_isotropy(
    aniso_ds,
    time_idx: int = 0,
    height_idx: int = 0,
    freq_step: int = 1,
    scatter_size: int = 35,
    show_path: bool = True,
    annotate_ends: bool = True,
    ax: plt.Axes = None,
):
    """
    Plot the barycentric triangle and the 'return to isotropy' path across
    freq_cutoff for a given (time, height). Each point is colored using
    the Emori–Jaccarino RGB triplet contained in aniso_ds['RGB'].

    Parameters
    ----------
    aniso_ds : xr.Dataset
        Output of anisotropy_pointwise(...) or anisotropy_from_ogives(...),
        exposing variables:
          - 'bary_x' (time, heights, freq_cutoff)
          - 'bary_y' (time, heights, freq_cutoff)
          - 'RGB'    (time, heights, freq_cutoff, rgb=3)
    time_idx : int
        Time index to plot.
    height_idx : int
        Height index to plot.
    freq_step : int
        Downsampling step along freq_cutoff for clarity (>=1).
    scatter_size : int
        Marker size for the colored points.
    show_path : bool
        If True, also draw a polyline connecting the points (order of freq_cutoff).
    annotate_ends : bool
        If True, annotate first/last cutoff points.
    ax : matplotlib.axes.Axes or None
        Provide an existing axes to draw on; if None, a new fig/ax is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    # --- Extract data
    bx = aniso_ds['bary_x'].isel(time=time_idx, heights=height_idx).values
    by = aniso_ds['bary_y'].isel(time=time_idx, heights=height_idx).values
    rgb = aniso_ds['RGB'].isel(time=time_idx, heights=height_idx).values  # shape (F, 3)

    # Safety checks and downsampling
    F = bx.shape[0]
    sel = slice(0, F, max(int(freq_step), 1))
    bx, by, rgb = bx[sel], by[sel], np.asarray(rgb[sel])

    # Clip RGB into [0,1] just in case
    rgb = np.clip(rgb, 0.0, 1.0)

    # --- Prepare axes
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created = True

    ax.set_aspect('equal', adjustable='box')

    # --- Barycentric triangle (corners: 1C, 2C, 3C)
    # 1C: (1, 0), 2C: (0, 0), 3C: (0.5, sqrt(3)/2)
    tri_x = np.array([1.0, 0.0, 0.5, 1.0])
    tri_y = np.array([0.0, 0.0, np.sqrt(3.0) / 2.0, 0.0])
    ax.plot(tri_x, tri_y, linewidth=1.5)

    # optional grid/edges inside (equilateral aesthetics)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, np.sqrt(3.0)/2.0 + 0.05)

    # Corner labels
    ax.text(1.0, 0.0, "1C", va='top', ha='left')
    ax.text(0.0, 0.0, "2C", va='top', ha='right')
    ax.text(0.5, np.sqrt(3.0)/2.0, "3C (isotropic)", va='bottom', ha='center')

    ax.set_xlabel("Barycentric coordinate $B_x$")
    ax.set_ylabel("Barycentric coordinate $B_y$")
    ax.set_title("Return to Isotropy in Barycentric Map")

    # --- Path and colored scatter
    if show_path:
        ax.plot(bx, by, linewidth=1.0)

    ax.scatter(bx, by, s=scatter_size, c=rgb)

    # --- Annotate start/end along freq_cutoff
    if annotate_ends and len(bx) >= 1:
        ax.annotate("low cutoff", (bx[0], by[0]), xytext=(5, 5), textcoords='offset points')
        ax.annotate("high cutoff", (bx[-1], by[-1]), xytext=(5, -12), textcoords='offset points')

        # small arrow indicating direction (from low to high cutoff)
        ax.annotate(
            "",
            xy=(bx[-1], by[-1]),
            xytext=(bx[0], by[0]),
            arrowprops=dict(arrowstyle="->", lw=1)
        )

    if created:
        plt.tight_layout()

    return ax

# Dopo aver creato il dataset di anisotropia:
# aniso_ds = anisotropy_pointwise(results["ogive"], return_rotation=True)

_ = plot_return_to_isotropy(
    ds_a,
    time_idx=19,
    height_idx=1,
    freq_step=2,        # opzionale: plotta 1 punto ogni 2 cutoff per pulizia
    scatter_size=30,
    show_path=True,
    annotate_ends=True
)

#%%
# --- imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from numpy import ma
import numpy as np
from scipy.linalg import eig
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)

def Anisotropy(Dataset, return_rotation = False, one_sonic = False):

    '''Works with xarray dataset with Reynolds stress tensor labels:
    #  uu,vv,ww,ecc..
    maybe problem with 1 height only meas
    returns the barycentric coordinates and RGB values
    returns isotropic data where nans are found
    '''
    
    #Filter nans to value 0 beacuse not handled by linalg.eig
    Dataset = Dataset.where(np.isfinite(Dataset.uu), other=0)
    if not one_sonic:
        (length,levels) = np.shape(Dataset['uu'])
    
        #Compute Reynolds tensor
        Reynolds_tensor = np.zeros((3,3,length,levels))
        trace=Dataset['uu'] + Dataset['vv'] + Dataset['ww']
    
        Reynolds_tensor[0,0] = Dataset['uu'].data/ma.masked_equal(trace,0) - 1/3
        Reynolds_tensor[0,1] = Dataset['uv'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[1,0] = Dataset['uv'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[1,1] = Dataset['vv'].data/ma.masked_equal(trace,0) - 1/3
        Reynolds_tensor[1,2] = Dataset['vw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[2,1] = Dataset['vw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[0,2] = Dataset['uw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[2,0] = Dataset['uw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[2,2] = Dataset['ww'].data/ma.masked_equal(trace,0) - 1/3
        
        #Compute eigenvalues
        eigenvalues = np.zeros((3,length,levels))
        eigenvectors = np.zeros((3,3, length, levels))
        for n in range(length):
            for l in range(0,levels):
                #compute
                eigval, eigvec = eig(Reynolds_tensor[:,:,n,l])
                #sort
                idx = eigval.argsort()[::-1]
                eigenvalues[:,n,l] = eigval[idx]
                eigenvectors[:, :, n, l] = eigvec[:, idx]
        
        #choose system of eigenvectors (not necessary)
        eigenvectors[:,0] = np.where(eigenvectors[0,0]>0, eigenvectors[:,0], -eigenvectors[:,0])
        eigenvectors[:,2] = np.where(eigenvectors[2,2]>0, eigenvectors[:,2], -eigenvectors[:,2])
        eigenvectors[:,1] = np.cross(eigenvectors[:,2],eigenvectors[:,0],axis=0)
    
        #check that the sum of eigenvalues is 0
        sum_eig=np.sum(eigenvalues)
        if  sum_eig > 0.000001 :
            print("Warning! the sum of all the eigenvalues is {}".format(sum_eig))
            
        #Compute barycentric map
        barycentric=np.zeros((2,length,levels))
        barycentric[0] = eigenvalues[0] - eigenvalues[1] + 1.5*eigenvalues[2] + 0.5
        barycentric[1] = np.sqrt(3)/2*(3*eigenvalues[2]+1)
        
        #Color code (Emori and Jaccarino 2014)
        RGB=np.zeros((3,length,levels))
        RGB[0]=eigenvalues[0]-eigenvalues[1]
        RGB[2]=2*(eigenvalues[1]-eigenvalues[2])
        RGB[1]=3*eigenvalues[2]+1
        RGB=np.moveaxis(RGB,0,-1)

    else:
        
        length = len(Dataset['uu'])
    
        #Compute Reynolds tensor
        Reynolds_tensor = np.zeros((3,3,length))
        trace=Dataset['uu'] + Dataset['vv'] + Dataset['ww']
    
        Reynolds_tensor[0,0] = Dataset['uu'].data/ma.masked_equal(trace,0) - 1/3
        Reynolds_tensor[0,1] = Dataset['uv'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[1,0] = Dataset['uv'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[1,1] = Dataset['vv'].data/ma.masked_equal(trace,0) - 1/3
        Reynolds_tensor[1,2] = Dataset['vw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[2,1] = Dataset['vw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[0,2] = Dataset['uw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[2,0] = Dataset['uw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[2,2] = Dataset['ww'].data/ma.masked_equal(trace,0) - 1/3
        
        #Compute eigenvalues
        eigenvalues = np.zeros((3,length))
        eigenvectors = np.zeros((3,3, length))
        for n in range(length):
            #compute
            eigval, eigvec = eig(Reynolds_tensor[:,:,n])
            #sort
            idx = eigval.argsort()[::-1]
            eigenvalues[:,n] = eigval[idx]
            eigenvectors[:, :, n] = eigvec[:, idx]
        
        #choose system of eigenvectors (not necessary)
        eigenvectors[:,0] = np.where(eigenvectors[0,0]>0, eigenvectors[:,0], -eigenvectors[:,0])
        eigenvectors[:,2] = np.where(eigenvectors[2,2]>0, eigenvectors[:,2], -eigenvectors[:,2])
        eigenvectors[:,1] = np.cross(eigenvectors[:,2],eigenvectors[:,0],axis=0)
    
        #check that the sum of eigenvalues is 0
        sum_eig=np.sum(eigenvalues)
        if  sum_eig > 0.000001 :
            print("Warning! the sum of all the eigenvalues is {}".format(sum_eig))
            
        #Compute barycentric map
        barycentric=np.zeros((2,length))
        barycentric[0] = eigenvalues[0] - eigenvalues[1] + 1.5*eigenvalues[2] + 0.5
        barycentric[1] = np.sqrt(3)/2*(3*eigenvalues[2]+1)
        
        #Color code (Emori and Jaccarino 2014)
        RGB=np.zeros((3,length))
        RGB[0]=eigenvalues[0]-eigenvalues[1]
        RGB[2]=2*(eigenvalues[1]-eigenvalues[2])
        RGB[1]=3*eigenvalues[2]+1
        RGB=np.moveaxis(RGB,0,-1)
    
    if return_rotation:
        return barycentric, RGB, eigenvalues, eigenvectors
    return barycentric, RGB

def compute_curve_with_original_anisotropy(ogive: xr.Dataset, time_idx: int = 0, height_idx: int = 0):
    """
    Applica la Anisotropy originale per ogni freq_cutoff e restituisce
    Bx, By, RGB lungo la frequenza per il punto (time_idx, height_idx).

    ogive: xr.Dataset con variabili 'uu','vv','ww','uv','uw','vw' e dims
           ('time','heights','freq_cutoff')
    """
    required = ['uu','vv','ww','uv','uw','vw']
    missing = [v for v in required if v not in ogive]
    if missing:
        raise ValueError(f"Mancano variabili in ogive: {missing}")

    T = ogive.sizes.get('time', 1)
    H = ogive.sizes.get('heights', 1)
    F = ogive.sizes.get('freq_cutoff', 1)

    Bx = np.full(F, np.nan, dtype=float)
    By = np.full(F, np.nan, dtype=float)
    RGB = np.full((F, 3), np.nan, dtype=float)

    for k in range(F):
        # prendi il dataset 2D (time × heights) a cutoff fisso
        ds2d = ogive[required].isel(freq_cutoff=k).transpose('time', 'heights', missing_dims='ignore')

        # applica la funzione originale (no clipping / no proiezioni)
        bary, rgb = Anisotropy(ds2d, return_rotation=False, one_sonic=False)

        # estrai il punto (t,h)
        Bx[k]  = float(bary[0, time_idx, height_idx])
        By[k]  = float(bary[1, time_idx, height_idx])
        RGB[k] = np.asarray(rgb[time_idx, height_idx, :], dtype=float)

    # clip dei colori a [0,1] solo per plotting (valori fuori non cambiano le coordinate!)
    RGB = np.clip(RGB, 0.0, 1.0)
    return Bx, By, RGB


def plot_barycentric_triangle(ax=None, title="Barycentric Map"):
    """Disegna il triangolo baricentrico e ritorna l'axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='box')

    tri_x = np.array([1.0, 0.0, 0.5, 1.0])
    tri_y = np.array([0.0, 0.0, np.sqrt(3.0)/2.0, 0.0])
    ax.plot(tri_x, tri_y, linewidth=1.5, color='k')

    ax.text(1.0, 0.0, "1C", va='top', ha='left')
    ax.text(0.0, 0.0, "2C", va='top', ha='right')
    ax.text(0.5, np.sqrt(3.0)/2.0, "3C (isotropic)", va='bottom', ha='center')

    # limiti un po' più larghi per vedere eventuali punti fuori
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, np.sqrt(3.0)/2.0 + 0.2)

    ax.set_xlabel("$B_x$")
    ax.set_ylabel("$B_y$")
    ax.set_title(title)
    return ax


def plot_return_to_isotropy_original(ogive: xr.Dataset, time_idx=0, height_idx=0,
                                     freq_step: int = 1, scatter_size: int = 35,
                                     annotate_ends: bool = True, ax=None):
    """
    Costruisce e plotta la curva 'return to isotropy' usando la funzione
    Anisotropy ORIGINALE (senza nessun clipping).
    """
    Bx, By, RGB = compute_curve_with_original_anisotropy(ogive, time_idx, height_idx)

    # downsample opzionale lungo cutoff
    sel = slice(0, len(Bx), max(int(freq_step), 1))
    Bx, By, RGB = Bx[sel], By[sel], RGB[sel]

    ax = plot_barycentric_triangle(ax=ax, title="Return to Isotropy (Original)")
    ax.plot(Bx, By, linewidth=1.0)               # path
    ax.scatter(Bx, By, s=scatter_size, c=RGB)     # punti colorati

    if annotate_ends and len(Bx) >= 1:
        ax.annotate("low cutoff", (Bx[0], By[0]), xytext=(5, 5), textcoords='offset points')
        ax.annotate("high cutoff", (Bx[-1], By[-1]), xytext=(5, -12), textcoords='offset points')
        ax.annotate("", xy=(Bx[-1], By[-1]), xytext=(Bx[0], By[0]),
                    arrowprops=dict(arrowstyle="->", lw=1))
    plt.tight_layout()
    return ax

#%%

# ====== ESEMPIO D'USO ======

ax = plot_return_to_isotropy_original(ogive, time_idx=2, height_idx=0, freq_step=1)
plt.show()
#%%
# ===========================
# Imports
# ===========================
import numpy as np
from numpy import ma
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)

import xarray as xr
from scipy.linalg import eig
import matplotlib.pyplot as plt


# ===========================
# Original anisotropy function
# ===========================
def Anisotropy(Dataset, return_rotation=False, one_sonic=False):
    """
    Compute anisotropy from a Reynolds stress tensor dataset.
    Returns barycentric coordinates and RGB color coding.
    """
    # Filter nans to zero
    Dataset = Dataset.where(np.isfinite(Dataset.uu), other=0)

    if not one_sonic:
        (length, levels) = np.shape(Dataset['uu'])
        Reynolds_tensor = np.zeros((3,3,length,levels))
        trace = Dataset['uu'] + Dataset['vv'] + Dataset['ww']

        Reynolds_tensor[0,0] = Dataset['uu'].data/ma.masked_equal(trace,0) - 1/3
        Reynolds_tensor[0,1] = Dataset['uv'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[1,0] = Dataset['uv'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[1,1] = Dataset['vv'].data/ma.masked_equal(trace,0) - 1/3
        Reynolds_tensor[1,2] = Dataset['vw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[2,1] = Dataset['vw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[0,2] = Dataset['uw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[2,0] = Dataset['uw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[2,2] = Dataset['ww'].data/ma.masked_equal(trace,0) - 1/3

        eigenvalues = np.zeros((3,length,levels))
        eigenvectors = np.zeros((3,3,length,levels))
        for n in range(length):
            for l in range(levels):
                eigval, eigvec = eig(Reynolds_tensor[:,:,n,l])
                idx = eigval.argsort()[::-1]
                eigenvalues[:,n,l] = eigval[idx]
                eigenvectors[:,:,n,l] = eigvec[:,idx]

        eigenvectors[:,0] = np.where(eigenvectors[0,0]>0, eigenvectors[:,0], -eigenvectors[:,0])
        eigenvectors[:,2] = np.where(eigenvectors[2,2]>0, eigenvectors[:,2], -eigenvectors[:,2])
        eigenvectors[:,1] = np.cross(eigenvectors[:,2],eigenvectors[:,0],axis=0)

        barycentric = np.zeros((2,length,levels))
        barycentric[0] = eigenvalues[0] - eigenvalues[1] + 1.5*eigenvalues[2] + 0.5
        barycentric[1] = np.sqrt(3)/2*(3*eigenvalues[2]+1)

        RGB = np.zeros((3,length,levels))
        RGB[0] = eigenvalues[0]-eigenvalues[1]
        RGB[2] = 2*(eigenvalues[1]-eigenvalues[2])
        RGB[1] = 3*eigenvalues[2]+1
        RGB = np.moveaxis(RGB,0,-1)

    else:
        length = len(Dataset['uu'])
        Reynolds_tensor = np.zeros((3,3,length))
        trace = Dataset['uu'] + Dataset['vv'] + Dataset['ww']

        Reynolds_tensor[0,0] = Dataset['uu'].data/ma.masked_equal(trace,0) - 1/3
        Reynolds_tensor[0,1] = Dataset['uv'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[1,0] = Dataset['uv'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[1,1] = Dataset['vv'].data/ma.masked_equal(trace,0) - 1/3
        Reynolds_tensor[1,2] = Dataset['vw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[2,1] = Dataset['vw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[0,2] = Dataset['uw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[2,0] = Dataset['uw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[2,2] = Dataset['ww'].data/ma.masked_equal(trace,0) - 1/3

        eigenvalues = np.zeros((3,length))
        eigenvectors = np.zeros((3,3,length))
        for n in range(length):
            eigval, eigvec = eig(Reynolds_tensor[:,:,n])
            idx = eigval.argsort()[::-1]
            eigenvalues[:,n] = eigval[idx]
            eigenvectors[:,:,n] = eigvec[:,idx]

        eigenvectors[:,0] = np.where(eigenvectors[0,0]>0, eigenvectors[:,0], -eigenvectors[:,0])
        eigenvectors[:,2] = np.where(eigenvectors[2,2]>0, eigenvectors[:,2], -eigenvectors[:,2])
        eigenvectors[:,1] = np.cross(eigenvectors[:,2],eigenvectors[:,0],axis=0)

        barycentric = np.zeros((2,length))
        barycentric[0] = eigenvalues[0] - eigenvalues[1] + 1.5*eigenvalues[2] + 0.5
        barycentric[1] = np.sqrt(3)/2*(3*eigenvalues[2]+1)

        RGB = np.zeros((3,length))
        RGB[0] = eigenvalues[0]-eigenvalues[1]
        RGB[2] = 2*(eigenvalues[1]-eigenvalues[2])
        RGB[1] = 3*eigenvalues[2]+1
        RGB = np.moveaxis(RGB,0,-1)

    if return_rotation:
        return barycentric, RGB, eigenvalues, eigenvectors
    return barycentric, RGB


# ===========================
# Wrapper: anisotropy at total flux
# ===========================
def anisotropy_total_flux(ogive: xr.Dataset, return_rotation: bool = False) -> xr.Dataset:
    """
    Compute anisotropy at the lowest freq_cutoff (≈ total flux).

    Parameters
    ----------
    ogive : xr.Dataset
        Output of compute_ogive, must contain uu, vv, ww, uv, uw, vw.
    return_rotation : bool
        If True, include eigenvalues and eigenvectors.

    Returns
    -------
    xr.Dataset
    """
    required = ['uu','vv','ww','uv','uw','vw']
    for v in required:
        if v not in ogive:
            raise ValueError(f"Missing variable {v} in ogive dataset")

    # lowest freq_cutoff
    f = ogive['freq_cutoff'].values
    k0 = int(np.argmin(f))
    ds2d = ogive[required].isel(freq_cutoff=k0).transpose('time','heights', missing_dims='ignore')

    out = Anisotropy(ds2d, return_rotation=return_rotation, one_sonic=False)
    if return_rotation:
        bary, rgb, e_vals, e_vecs = out
    else:
        bary, rgb = out

    T = ds2d.sizes.get('time', 1)
    H = ds2d.sizes.get('heights', 1)

    coords = {'time': ds2d['time'], 'heights': ds2d['heights'], 'rgb':[0,1,2]}
    data_vars = {
        'bary_x': (('time','heights'), np.asarray(bary[0]).reshape(T,H)),
        'bary_y': (('time','heights'), np.asarray(bary[1]).reshape(T,H)),
        'RGB':    (('time','heights','rgb'), np.asarray(rgb).reshape(T,H,3)),
    }
    if return_rotation:
        coords.update({'eig':[0,1,2], 'axis3':[0,1,2]})
        data_vars.update({
            'eigvals': (('eig','time','heights'), np.asarray(e_vals).reshape(3,T,H)),
            'eigvecs': (('eig','axis3','time','heights'), np.asarray(e_vecs).reshape(3,3,T,H)),
        })

    ds_tot = xr.Dataset(data_vars, coords=coords)
    return ds_tot


# ===========================
# Plotting function
# ===========================
def plot_barycentric_total(aniso_total: xr.Dataset,
                           time_idx: int = 0,
                           height_idx: int = 0,
                           ax=None):
    """
    Plot barycentric anisotropy point in Lumley triangle (total flux).
    """
    bx = float(aniso_total['bary_x'].isel(time=time_idx, heights=height_idx))
    by = float(aniso_total['bary_y'].isel(time=time_idx, heights=height_idx))
    rgb = np.clip(aniso_total['RGB'].isel(time=time_idx, heights=height_idx).values, 0, 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    ax.set_aspect('equal', adjustable='box')

    tri_x = [1.0, 0.0, 0.5, 1.0]
    tri_y = [0.0, 0.0, np.sqrt(3)/2.0, 0.0]
    ax.plot(tri_x, tri_y, color='k')

    ax.text(1.0, 0.0, "1C", va='top', ha='left')
    ax.text(0.0, 0.0, "2C", va='top', ha='right')
    ax.text(0.5, np.sqrt(3)/2.0, "3C", va='bottom', ha='center')

    ax.scatter([bx], [by], s=100, c=[rgb], edgecolors='k', zorder=3)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, np.sqrt(3)/2.0 + 0.05)
    ax.set_xlabel("$B_x$")
    ax.set_ylabel("$B_y$")
    ax.set_title(f"Total Flux Anisotropy (t={time_idx}, h={height_idx})")

    plt.tight_layout()
    return ax

def plot_barycentric_all(aniso_total: xr.Dataset,
                         height_idx: int = 0,
                         ax=None):
    """
    Plot barycentric anisotropy points for ALL time steps
    at a given height inside the Lumley triangle.

    Parameters
    ----------
    aniso_total : xr.Dataset
        Output from anisotropy_total_flux (must contain bary_x, bary_y, RGB).
    height_idx : int
        Height index to select.
    ax : matplotlib axis, optional
        Axis to plot on. If None, a new figure is created.
    """
    bx = aniso_total['bary_x'].isel(heights=height_idx).values
    by = aniso_total['bary_y'].isel(heights=height_idx).values
    rgb = np.clip(aniso_total['RGB'].isel(heights=height_idx).values, 0, 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_aspect('equal', adjustable='box')

    # draw Lumley triangle
    tri_x = [1.0, 0.0, 0.5, 1.0]
    tri_y = [0.0, 0.0, np.sqrt(3)/2.0, 0.0]
    ax.plot(tri_x, tri_y, color='k')

    # labels
    ax.text(1.0, 0.0, "1C", va='top', ha='left')
    ax.text(0.0, 0.0, "2C", va='top', ha='right')
    ax.text(0.5, np.sqrt(3)/2.0, "3C", va='bottom', ha='center')

    # all points (time dimension)
    ax.scatter(bx, by, s=60, c=rgb, edgecolors='k', zorder=3)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, np.sqrt(3)/2.0 + 0.05)
    ax.set_xlabel("$B_x$")
    ax.set_ylabel("$B_y$")
    ax.set_title(f"Total Flux Anisotropy (all times, h={height_idx})")

    plt.tight_layout()
    return ax

# ===========================
# Example usage
# ===========================
# ogive_ds = compute_ogive(results["spectra"])

aniso_total = anisotropy_total_flux(ogive, return_rotation=True)
plot_barycentric_all(aniso_total, height_idx=0)
plt.show()
#%%
print(ogive)
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

def anisotropy_path(ogive_ds, time_idx=1, height_idx=0, ax=None):
    """
    Plot return-to-isotropy path (barycentric map) across freq_cutoff.

    Parameters
    ----------
    ogive_ds : xr.Dataset
        Ogive dataset with uu,vv,ww,uv,uw,vw vs freq_cutoff.
    time_idx : int
        Time index to select.
    height_idx : int
        Height index to select.
    ax : matplotlib axis, optional
        Axis to plot on. If None, a new one is created.

    Returns
    -------
    ax : matplotlib axis
    """

    # estraggo i dati lungo freq_cutoff
    uu = ogive_ds['uu'].isel(time=time_idx, heights=height_idx).values
    vv = ogive_ds['vv'].isel(time=time_idx, heights=height_idx).values
    ww = ogive_ds['ww'].isel(time=time_idx, heights=height_idx).values
    uv = ogive_ds['uv'].isel(time=time_idx, heights=height_idx).values
    uw = ogive_ds['uw'].isel(time=time_idx, heights=height_idx).values
    vw = ogive_ds['vw'].isel(time=time_idx, heights=height_idx).values

    bary_x, bary_y = [], []

    for i in range(len(ogive_ds.freq_cutoff)):
        R = np.array([[uu[i], uv[i], uw[i]],
                      [uv[i], vv[i], vw[i]],
                      [uw[i], vw[i], ww[i]]])

        trace = np.trace(R)
        if trace <= 0:  # skip cutoff non fisici
            bary_x.append(np.nan)
            bary_y.append(np.nan)
            continue

        # tensor normalizzato e shiftato
        aij = R/trace - np.eye(3)/3
        eigvals, _ = eig(aij)
        eigvals = np.real(np.sort(eigvals)[::-1])  # ordina discendente

        # barycentric coordinates
        bx = eigvals[0] - eigvals[1] + 1.5*eigvals[2] + 0.5
        by = np.sqrt(3)/2 * (3*eigvals[2] + 1)

        bary_x.append(bx)
        bary_y.append(by)

    bary_x, bary_y = np.array(bary_x), np.array(bary_y)

    # plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')

    # triangolo Lumley
    tri_x = [1,0,0.5,1]
    tri_y = [0,0,np.sqrt(3)/2,0]
    ax.plot(tri_x, tri_y, 'k-')

    # curve anisotropia
    ax.plot(bary_x, bary_y, marker='o', ms=2, lw=1, label=f"time={time_idx}, h={height_idx}")
    ax.legend()
    ax.set_xlabel("$B_x$")
    ax.set_ylabel("$B_y$")
    ax.set_title("Return to isotropy path")

    return ax

# un esempio su time=0, height=0
anisotropy_path(ogive, time_idx=100, height_idx=1)
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt

def plot_barycentric_triangle(ax=None, perc=0.7, fill_zones=True):
    """
    Plot the Lumley barycentric triangle with optional asymptotic zones (1C, 2C, 3C).

    Parameters
    ----------
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates a new one.
    perc : float
        Fraction used for asymptotic zone triangles (default=0.7).
    fill_zones : bool
        If True, fill asymptotic subzones (1C, 2C, 3C).
    """
    # main triangle points
    A = np.array([0, 0])
    C = np.array([0.5, np.sqrt(3)/2])
    E = np.array([1, 0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    # plot main triangle
    tri = np.array([A, C, E, A])
    ax.plot(tri[:,0], tri[:,1], 'k-', lw=1.5)

    # optional fill of asymptotic zones
    if fill_zones:
        from matplotlib.patches import Polygon
        # reuse your triangles() function
        def triangles(state, perc):
            bary = np.array([0.5, np.sqrt(3)/6])
            A = np.array([0, 0])
            B = np.array([0.5*np.cos(np.pi/180*60), 0.5*np.sin(np.pi/180*60)])
            C = np.array([0.5, np.sqrt(3)/2])
            D = np.array([1 - B[0], B[1]])
            E = np.array([1, 0])
            F = np.array([0.5, 0])
            if state == '1c':
                bary2 = [1 - bary[0]*perc, bary[1]*perc]
                triangle1 = [E, [1 - B[0]*perc, D[1]*perc], bary2]
                triangle2 = [E, F*(2 - perc), bary2]
            elif state == '2c':
                bary2 = bary * perc
                triangle1 = [A, B * perc, bary2]
                triangle2 = [A, F * perc, bary2]
            elif state == '3c':
                bary2 = [bary[0], C[1] - (C[1] - bary[1]) * perc]
                triangle1 = [C, B * (2 - perc), bary2]
                triangle2 = [C, [1 - B[0] * (2 - perc), D[1] * (2 - perc)], bary2]
            return np.array(triangle1), np.array(triangle2)

        colors = {"1c":"#ff9999","2c":"#99ccff","3c":"#99ff99"}
        for state in ["1c","2c","3c"]:
            t1,t2 = triangles(state, perc)
            poly1 = Polygon(t1, closed=True, facecolor=colors[state], alpha=0.3)
            poly2 = Polygon(t2, closed=True, facecolor=colors[state], alpha=0.3)
            ax.add_patch(poly1)
            ax.add_patch(poly2)

    ax.set_aspect("equal")
    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,np.sqrt(3)/2+0.1)
    ax.set_xlabel("$B_x$")
    ax.set_ylabel("$B_y$")
    ax.set_title("Barycentric triangle (with asymptotic zones)")

    return ax


ax = plot_barycentric_triangle(perc=0.7, fill_zones=True)
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from matplotlib.patches import Polygon

def plot_barycentric_triangle(ax, perc=0.7, fill_zones=True):
    """Draw background Lumley barycentric triangle with asymptotic zones."""
    # main triangle vertices
    A = np.array([0, 0])
    C = np.array([0.5, np.sqrt(3)/2])
    E = np.array([1, 0])
    tri = np.array([A, C, E, A])
    ax.plot(tri[:,0], tri[:,1], 'k-', lw=1.5)

    if fill_zones:
        def triangles(state, perc):
            bary = np.array([0.5, np.sqrt(3)/6])
            A = np.array([0, 0])
            B = np.array([0.5*np.cos(np.pi/180*60), 0.5*np.sin(np.pi/180*60)])
            C = np.array([0.5, np.sqrt(3)/2])
            D = np.array([1 - B[0], B[1]])
            E = np.array([1, 0])
            F = np.array([0.5, 0])
            if state == '1c':
                bary2 = [1 - bary[0]*perc, bary[1]*perc]
                triangle1 = [E, [1 - B[0]*perc, D[1]*perc], bary2]
                triangle2 = [E, F*(2 - perc), bary2]
            elif state == '2c':
                bary2 = bary * perc
                triangle1 = [A, B * perc, bary2]
                triangle2 = [A, F * perc, bary2]
            elif state == '3c':
                bary2 = [bary[0], C[1] - (C[1] - bary[1]) * perc]
                triangle1 = [C, B * (2 - perc), bary2]
                triangle2 = [C, [1 - B[0] * (2 - perc), D[1] * (2 - perc)], bary2]
            return np.array(triangle1), np.array(triangle2)

        colors = {"1c":"#ff9999","2c":"#99ccff","3c":"#99ff99"}
        for state in ["1c","2c","3c"]:
            t1,t2 = triangles(state, perc)
            poly1 = Polygon(t1, closed=True, facecolor=colors[state], alpha=0.3)
            poly2 = Polygon(t2, closed=True, facecolor=colors[state], alpha=0.3)
            ax.add_patch(poly1)
            ax.add_patch(poly2)

    ax.set_aspect("equal")
    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,np.sqrt(3)/2+0.1)
    ax.set_xlabel("$B_x$")
    ax.set_ylabel("$B_y$")
    ax.set_title("Return to isotropy path")
    return ax


def anisotropy_path(ogive_ds, time_idx=0, height_idx=0, perc=0.7, fill_zones=True, ax=None):
    """
    Plot return-to-isotropy path (barycentric map) across freq_cutoff.

    Parameters
    ----------
    ogive_ds : xr.Dataset
        Ogive dataset with uu,vv,ww,uv,uw,vw vs freq_cutoff.
    time_idx : int
        Time index to select.
    height_idx : int
        Height index to select.
    perc : float
        Percentage used for asymptotic zone shading.
    fill_zones : bool
        Whether to fill 1C/2C/3C zones.
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates a new one.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    # draw background triangle
    plot_barycentric_triangle(ax, perc=perc, fill_zones=fill_zones)

    # estraggo i dati lungo freq_cutoff
    uu = ogive_ds['uu'].isel(time=time_idx, heights=height_idx).values
    vv = ogive_ds['vv'].isel(time=time_idx, heights=height_idx).values
    ww = ogive_ds['ww'].isel(time=time_idx, heights=height_idx).values
    uv = ogive_ds['uv'].isel(time=time_idx, heights=height_idx).values
    uw = ogive_ds['uw'].isel(time=time_idx, heights=height_idx).values
    vw = ogive_ds['vw'].isel(time=time_idx, heights=height_idx).values

    bary_x, bary_y = [], []

    for i in range(len(ogive_ds.freq_cutoff)):
        R = np.array([[uu[i], uv[i], uw[i]],
                      [uv[i], vv[i], vw[i]],
                      [uw[i], vw[i], ww[i]]])
        trace = np.trace(R)
        if trace <= 0:
            bary_x.append(np.nan)
            bary_y.append(np.nan)
            continue
        aij = R/trace - np.eye(3)/3
        eigvals, _ = eig(aij)
        eigvals = np.real(np.sort(eigvals)[::-1])
        bx = eigvals[0] - eigvals[1] + 1.5*eigvals[2] + 0.5
        by = np.sqrt(3)/2 * (3*eigvals[2] + 1)
        bary_x.append(bx)
        bary_y.append(by)

    bary_x, bary_y = np.array(bary_x), np.array(bary_y)

    # plot anisotropy path
    ax.plot(bary_x, bary_y, marker='o', ms=2, lw=1, label=f"time={time_idx}, h={height_idx}")
    ax.legend()
    return ax

anisotropy_path(ogive, time_idx=7, height_idx=1, perc=0.7, fill_zones=True)
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

def reynolds_from_spectra(spectra, time_idx=0, height_idx=0):
    """Compute Reynolds stresses by integrating spectra directly."""
    f = spectra['freq'].values
    su  = spectra['su'].isel(time=time_idx, heights=height_idx).values
    sv  = spectra['sv'].isel(time=time_idx, heights=height_idx).values
    sw  = spectra['sw'].isel(time=time_idx, heights=height_idx).values
    cuv = spectra['cuv'].isel(time=time_idx, heights=height_idx).values
    cuw = spectra['cuw'].isel(time=time_idx, heights=height_idx).values
    cvw = spectra['cvw'].isel(time=time_idx, heights=height_idx).values

    uu = np.trapz(su, f)
    vv = np.trapz(sv, f)
    ww = np.trapz(sw, f)
    uv = np.trapz(cuv, f)
    uw = np.trapz(cuw, f)
    vw = np.trapz(cvw, f)
    return uu, vv, ww, uv, uw, vw

def reynolds_from_ogive(ogive, time_idx=0, height_idx=0):
    """Take last cutoff from ogive dataset as Reynolds stress."""
    uu = ogive['uu'].isel(time=time_idx, heights=height_idx, freq_cutoff=0).values
    vv = ogive['vv'].isel(time=time_idx, heights=height_idx, freq_cutoff=0).values
    ww = ogive['ww'].isel(time=time_idx, heights=height_idx, freq_cutoff=0).values
    uv = ogive['uv'].isel(time=time_idx, heights=height_idx, freq_cutoff=0).values
    uw = ogive['uw'].isel(time=time_idx, heights=height_idx, freq_cutoff=0).values
    vw = ogive['vw'].isel(time=time_idx, heights=height_idx, freq_cutoff=0).values
    return uu, vv, ww, uv, uw, vw

def barycentric_coords(R):
    """Compute barycentric coordinates from Reynolds stress tensor."""
    trace = np.trace(R)
    aij = R/trace - np.eye(3)/3
    eigvals, _ = eig(aij)
    eigvals = np.real(np.sort(eigvals)[::-1])  # λ1 ≥ λ2 ≥ λ3
    bx = eigvals[0] - eigvals[1] + 1.5*eigvals[2] + 0.5
    by = np.sqrt(3)/2*(3*eigvals[2]+1)
    return bx, by, eigvals

# Example usage:
time_idx, height_idx = 0, 0

# from spectra
uu, vv, ww, uv, uw, vw = reynolds_from_spectra(ds['spectra'], time_idx, height_idx)
R_spec = np.array([[uu, uv, uw],[uv, vv, vw],[uw, vw, ww]])
bx_spec, by_spec, eig_spec = barycentric_coords(R_spec)

# from ogive (lowest cutoff → full flux)
uu, vv, ww, uv, uw, vw = reynolds_from_ogive(ogive, time_idx, height_idx)
R_og = np.array([[uu, uv, uw],[uv, vv, vw],[uw, vw, ww]])
bx_og, by_og, eig_og = barycentric_coords(R_og)

# print comparison
print("Eigenvalues spectra:", eig_spec, "sum:", eig_spec.sum())
print("Eigenvalues ogive  :", eig_og, "sum:", eig_og.sum())
print("Bary coords spectra:", bx_spec, by_spec)
print("Bary coords ogive  :", bx_og, by_og)

# quick plot
plt.figure(figsize=(6,6))
plt.plot([0,1,0.5,0],[0,0,np.sqrt(3)/2,0],'k-')  # main triangle
plt.scatter(bx_spec, by_spec, c="blue", label="from spectra")
plt.scatter(bx_og, by_og, c="red", marker="x", s=80, label="from ogive")
plt.legend()
plt.xlabel("Bx")
plt.ylabel("By")
plt.title(f"Anisotropy check (time={time_idx}, height={height_idx})")
plt.axis("equal")
plt.show()
