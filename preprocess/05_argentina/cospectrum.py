#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import xgcm 
from fastjmd95 import jmd95numba 

case_name = '05_argentina'

# paths to dataset
PATH_2d = '/scratch/ab10313/pleiades/'+case_name+'/2d_data/'
PATH_3d = '/scratch/ab10313/pleiades/'+case_name+'/3d_data/'

# make diirectory for preprocessed variables
PATH_PP = '/scratch/ab10313/pleiades/'+case_name+'/preprcossed_data/'
#os.mkdir(PATH_PP)


# load 3d data
ds_T = xr.open_dataset(PATH_3d+'ds_T.nc',engine="h5netcdf")
ds_S = xr.open_dataset(PATH_3d+'ds_S.nc',engine="h5netcdf")
ds_W = xr.open_dataset(PATH_3d+'ds_W.nc',engine="h5netcdf")



# find min and max i and j to crop to 10X10 degrees

i_min = np.max([ds_T.i.min().values, ds_S.i.min().values, ds_W.i.min().values])
i_max = np.min([ds_T.i.max().values, ds_S.i.max().values, ds_W.i.max().values])
j_min = np.max([ds_T.j.min().values, ds_S.j.min().values, ds_W.j.min().values])
j_max = np.min([ds_T.j.max().values, ds_S.j.max().values, ds_W.j.max().values])


#define slice to 480 index

if i_min+480>i_max:
    print('cropped region error in i')
elif j_min+480>j_max:
    print('cropped region error in j')
else:
    i_slice = slice(i_min,i_min+480)
    j_slice = slice(j_min,j_min+480)



# merge datasets

ds_3d =xr.merge([ds_T.sel(i=i_slice,j=j_slice), ds_S.sel(i=i_slice,j=j_slice), ds_W.sel(i=i_slice,j=j_slice)])


# define grids 
grid_3d = xgcm.Grid(ds_3d)


# sigma from temp and salt, using the fastjmd95 package
    
# reference density 
rho0 = 1000 #kg/m^3

# potential density anomaly 
# with the reference pressure of 0 dbar and ρ0 = 1000 kg m−3
sigma0 = jmd95numba.rho(ds_3d.Salt.chunk(chunks={'time': 1, 'j': ds_3d.j.size, 'i': ds_3d.i.size}),
                         ds_3d.Theta.chunk(chunks={'time': 1, 'j': ds_3d.j.size, 'i': ds_3d.i.size}), 0) - rho0

# gravity
G = 9.81 #m/s^2

# buoyancy
B = -G*sigma0/rho0
B = B.rename('Buoyancy')


del sigma0


# interp W 
W_interp = grid_3d.interp(ds_3d.W,'Z', boundary='extend')

del grid_3d, ds_3d


# # cospectrum of w and b at the surface

B_drop = B.drop(['CS', 'SN', 'Depth', 'dxF', 'dyF', 'rA', 'XC', 'YC','hFacC']).fillna(0)
del B

#  spectra
import xrft
B_spectra = xrft.isotropic_power_spectrum(B_drop,dim=['i','j'], 
                                           detrend='linear', window=True).compute().mean('time')

#save spectra
B_spectra.to_netcdf(PATH_PP+'B_spectra.nc',engine='h5netcdf')
del B_drop


W_spectra = xrft.isotropic_power_spectrum(W_interp,dim=['i','j'], 
                                           detrend='linear', window=True).compute().mean('time')
#save spectra
W_spectra.to_netcdf(PATH_PP+'W_spectra.nc',engine='h5netcdf')
del W_interp



WB_cross_spectra = (W_spectra*np.conjugate(B_spectra)).real

#save spectra
WB_cross_spectra.to_netcdf(PATH_PP+'WB_cross_spectra.nc',engine='h5netcdf')


