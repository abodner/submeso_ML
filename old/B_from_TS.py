#!/usr/bin/env python
# coding: utf-8

from xmitgcm import llcreader
model = llcreader.ECCOPortalLLC4320Model()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import xrft
import xgcm
import gcm_filters
import matplotlib.pyplot as plt
import xesmf as xe
import gsw_xarray as gsw
import os
from datetime import datetime

CASE_NAME = 'agulhas/'
BASE = '/scratch/ab10313/submeso_ML_data/'
PATH = BASE+CASE_NAME

print('PREPROCESS CASE '+CASE_NAME)
print(datetime.now())

print('LOAD ds_T ds_S')
print(datetime.now())
# load temperature and salinity datasets
ds_T = xr.open_dataset(PATH+'raw_data/T.nc')
ds_S = xr.open_dataset(PATH+'raw_data/S.nc')


T = ds_T.Theta.chunk(chunks={'time': 1, 'k':T_prechunk.k.size, 'j': T_prechunk.j.size, 'i': T_prechunk.i.size})
S = ds_S.Salt.chunk(chunks={'time': 1, 'k':T_prechunk.k.size, 'j': T_prechunk.j.size, 'i': T_prechunk.i.size})

del ds_T, ds_S

print('CALCULATE B')
print(datetime.now())
# calculate buoyancy from temp and salt, using the gsw package
# pressure
P = gsw.p_from_z(T.Z, T.YC)

# gravity
G = gsw.grav(T.YC, P)

# Conservative Temperature (CT)
CT = gsw.CT_from_pt(S, T)

# potential density anomaly 
# with the reference pressure of 0 dbar and ρ0 = 1000 kg m−3
sigma0 = gsw.sigma0(S, CT)

# reference density 
rho0 = 1000 #kg/m^3

# buoyancy
B = -G*sigma0/rho0

B = B.rename('Buoyancy')

del S, P, G, CT


print('SAVE B')
print(datetime.now())

print('B:START')
print(datetime.now())
B.to_netcdf(PATH+'raw_data/B.nc',engine='h5netcdf')
print('B:COMPLETE')
print(datetime.now())





