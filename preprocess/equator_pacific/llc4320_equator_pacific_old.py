#!/usr/bin/env python
# coding: utf-8

from xmitgcm import llcreader
model = llcreader.ECCOPortalLLC4320Model()
import numpy as np
import pandas as pd
import xarray as xr
import os
from datetime import datetime

CASE_NAME = 'equator_pacific/'
BASE = '/scratch/ab10313/submeso_ML_data/'
PATH = BASE+CASE_NAME

print('PREPROCESS CASE '+CASE_NAME)
print(datetime.now())
isExist = os.path.exists(PATH)
if isExist:
    print('path already exsits...')
if not isExist:
    os.mkdir(PATH)

print('T,S,UV,W FULL DATASETS FROM LLC4320')
print(datetime.now())
# load full temperature and salinity datasets
ds_T_full = model.get_dataset(varnames=['Theta'], type='latlon')
ds_S_full = model.get_dataset(varnames=['Salt'], type='latlon')

# load full velocity dataset 
ds_UV_full = model.get_dataset(varnames=['U','V'], type='latlon')
ds_W_full = model.get_dataset(varnames=['W'], type='latlon')

# select subdomain area and downsample every 24 hours to reduce auto-correlation between samples (Uchida et. al. 2022)
# depth above -700m (as we will average over the mixed layer below)
# lat= -57:-42, lon=23:37 # change domain to be larger than 12 degrees

lat_min = -8
lat_max = 8
lon_min = -145
lon_max = -130
depth_lim = -700 

sel_area = np.logical_and(np.logical_and(np.logical_and(ds_T_full.XC>lon_min, ds_T_full.XC<lon_max ),
                           np.logical_and(ds_T_full.YC>lat_min, ds_T_full.YC<lat_max)), ds_T_full.Z>depth_lim)

sel_area_U = np.logical_and(np.logical_and(np.logical_and(ds_UV_full.XG.mean('j_g')>lon_min, ds_UV_full.XG.mean('j_g')<lon_max),
                           np.logical_and(ds_UV_full.YC.mean('i')>lat_min, ds_UV_full.YC.mean('i')<lat_max)),ds_UV_full.Z>depth_lim)

sel_area_V = np.logical_and(np.logical_and(np.logical_and(ds_UV_full.XC.mean('j')>lon_min, ds_UV_full.XC.mean('j')<lon_max),
                           np.logical_and(ds_UV_full.YG.mean('i_g')>lat_min, ds_UV_full.YG.mean('i_g')<lat_max)),ds_UV_full.Z>depth_lim)

sel_area_W = np.logical_and(np.logical_and(np.logical_and(ds_W_full.XC>lon_min, ds_W_full.XC<lon_max ),
                           np.logical_and(ds_W_full.YC>lat_min, ds_W_full.YC<lat_max)), ds_W_full.Zl>depth_lim)


# temp, salt and velocity in subdomain
ds_T = ds_T_full.where(sel_area, drop=True).resample(time='24H').nearest(tolerance="1H")
ds_S = ds_S_full.where(sel_area, drop=True).resample(time='24H').nearest(tolerance="1H")
ds_U = ds_UV_full.U.where(sel_area_U, drop=True).resample(time='24H').nearest(tolerance="1H")
ds_V = ds_UV_full.V.where(sel_area_V, drop=True).resample(time='24H').nearest(tolerance="1H")
ds_W = ds_W_full.where(sel_area_W, drop=True).resample(time='24H').nearest(tolerance="1H")

del ds_T_full, ds_S_full, ds_UV_full, ds_W_full


# SAVE NETCDF
print('SAVE NETCDF: ds_T,ds_S,ds_UV,ds_W')

isExist = os.path.exists(PATH+'raw_data/')
if isExist:
    print('path already exsits...')
if not isExist:
    os.mkdir(PATH+'raw_data/')

print('ds_T:START')
print(datetime.now())
ds_T.to_netcdf(PATH+'raw_data/ds_T.nc',engine='h5netcdf')
del ds_T
print('ds_T:COMPLETE')

print('ds_S:START')
print(datetime.now())
ds_S.to_netcdf(PATH+'raw_data/ds_S.nc',engine='h5netcdf')
del ds_S
print('ds_S:COMPLETE')

print('ds_U:START')
print(datetime.now())
ds_U.to_netcdf(PATH+'raw_data/ds_U.nc',engine='h5netcdf')
del ds_U
print('U:COMPLETE')

print('ds_V:START')
print(datetime.now())
ds_V.to_netcdf(PATH+'raw_data/ds_V.nc',engine='h5netcdf')
del ds_V
print('V:COMPLETE')

print('ds_W:START')
print(datetime.now())
ds_W.to_netcdf(PATH+'raw_data/ds_W.nc',engine='h5netcdf')
del ds_W
print('ds_W:COMPLETE')

