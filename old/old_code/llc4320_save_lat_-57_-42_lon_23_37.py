#!/usr/bin/env python
# coding: utf-8

from xmitgcm import llcreader
model = llcreader.ECCOPortalLLC4320Model()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import gsw_xarray as gsw
import os
from datetime import datetime

CASE_NAME = 'lat_-57_-42_lon_23_37/'
BASE = '/scratch/ab10313/submeso_ML_data/'
PATH = BASE+CASE_NAME

print('PREPROCESS CASE '+CASE_NAME)
print(datetime.now())

print('T,S,U,V,W FROM LLC4320')
print(datetime.now())
# load full temperature and salinity datasets
ds_T_full = model.get_dataset(varnames=['Theta'], type='latlon')
ds_S_full = model.get_dataset(varnames=['Salt'], type='latlon')

# load full velocity dataset 
ds_UV_full = model.get_dataset(varnames=['U','V'], type='latlon')
ds_W_full = model.get_dataset(varnames=['W'], type='latlon')

# interp full velocities onto tracer grid
ds_W_full_interp = ds_W_full.interp(k_l=ds_T_full.k)
ds_UV_full_interp = ds_UV_full.interp(j_g=ds_T_full.j, i_g=ds_T_full.i)

del ds_UV_full, ds_W_full


# select subdomain area and downsample every 24 hours to reduce auto-correlation between samples (Uchida et. al. 2022)
# depth above -1000m (as we will average over the mixed layer below)
# lat= 25:40, lon=-72:-58 # change domain to be larger than 12 degrees

lat_min = -57
lat_max = -42
lon_min = 23
lon_max = 37
depth_lim = -1000 

sel_area = np.logical_and(np.logical_and(np.logical_and(ds_T_full.XC>lon_min, ds_T_full.XC<lon_max ),
                           np.logical_and(ds_T_full.YC>lat_min, ds_T_full.YC<lat_max)), ds_T_full.Z>depth_lim)

# temp, salt and velocity in subdomain
T_prechunk = ds_T_full.Theta.resample(time='24H').nearest(tolerance="1H").where(sel_area, drop=True)
S_prechunk = ds_S_full.Salt.resample(time='24H').nearest(tolerance="1H").where(sel_area, drop=True)
U_prechunk = ds_UV_full_interp.U.resample(time='24H').nearest(tolerance="1H").where(sel_area, drop=True)
V_prechunk = ds_UV_full_interp.V.resample(time='24H').nearest(tolerance="1H").where(sel_area, drop=True)
W_prechunk = ds_W_full_interp.W.resample(time='24H').nearest(tolerance="1H").where(sel_area, drop=True)

del ds_T_full, ds_S_full, ds_UV_full_interp, ds_W_full_interp

T = T_prechunk.chunk(chunks={'time': 1, 'k':T_prechunk.k.size, 'j': T_prechunk.j.size, 'i': T_prechunk.i.size})
S = S_prechunk.chunk(chunks={'time': 1, 'k':S_prechunk.k.size, 'j': S_prechunk.j.size, 'i': S_prechunk.i.size})
U = U_prechunk.chunk(chunks={'time': 1, 'k':U_prechunk.k.size, 'j': U_prechunk.j.size, 'i': U_prechunk.i.size})
V = V_prechunk.chunk(chunks={'time': 1, 'k':V_prechunk.k.size, 'j': V_prechunk.j.size, 'i': V_prechunk.i.size})
W = W_prechunk.chunk(chunks={'time': 1, 'k':W_prechunk.k.size, 'j': W_prechunk.j.size, 'i': W_prechunk.i.size})

del T_prechunk, S_prechunk, U_prechunk, V_prechunk, W_prechunk

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


# SAVE NETCDF
print('SAVE NETCDF: B,U,V,W')
print('B')
B.to_netcdf(PATH+'data/B.nc',engine='h5netcdf')
print('U')
U.to_netcdf(PATH+'data/U.nc',engine='h5netcdf')
print('V')
V.to_netcdf(PATH+'data/V.nc',engine='h5netcdf')
print('W')
W.to_netcdf(PATH+'data/W.nc',engine='h5netcdf')


