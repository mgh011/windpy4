#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 08:26:18 2025

@author: mauro_ghirardelli
"""

def inspect_ds(ds):
    for key, sub in ds.items():
        print(f"\n=== {key} ===")
        if hasattr(sub, "dims"):  #is it an xarray.Dataset
            print(f"type: {type(sub)}")
            print(f"dims: {sub.dims}")
            print("variables:")
            for v in sub.data_vars:
                print(f"  - {v:15s} {sub[v].dims} {sub[v].shape}")
        else:
            print(f"{key}: tipo non xarray -> {type(sub)}")