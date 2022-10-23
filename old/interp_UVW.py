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

print('INTERPOLATE U,V,W FROM LLC4320 TO T GRID')
print(datetime.now())
# load temperature dataset
ds_T = xr.open_dataset(PATH+'raw_data/T.nc')

# load full velocity dataset 
ds_U = xr.open_dataset(PATH+'raw_data/U.nc')
ds_V = xr.open_dataset(PATH+'raw_data/V.nc')
ds_W = xr.open_dataset(PATH+'raw_data/W.nc')

# interp full velocities onto tracer grid
ds_U_interp = ds_U.interp(j_g=ds_T_full.j, i_g=ds_T_full.i)
ds_V_interp = ds_V.interp(j_g=ds_T_full.j, i_g=ds_T_full.i)
ds_W_interp = ds_W.interp(k_l=ds_T_full.k)

del ds_U, ds_V, ds_W


T = ds_T.Theta.chunk(chunks={'time': 1, 'k':T_prechunk.k.size, 'j': T_prechunk.j.size, 'i': T_prechunk.i.size})
S = ds_S.Salt.chunk(chunks={'time': 1, 'k':T_prechunk.k.size, 'j': T_prechunk.j.size, 'i': T_prechunk.i.size})
U = ds_U_interp.U.chunk(chunks={'time': 1, 'k':T_prechunk.k.size, 'j': T_prechunk.j.size, 'i': T_prechunk.i.size})
V = ds_V_interp.U.chunk(chunks={'time': 1, 'k':T_prechunk.k.size, 'j': T_prechunk.j.size, 'i': T_prechunk.i.size})
W = ds_W_interp.U.chunk(chunks={'time': 1, 'k':T_prechunk.k.size, 'j': T_prechunk.j.size, 'i': T_prechunk.i.size})

del ds_T, ds_S, ds_U_interp, ds_V_interp, ds_W_interp

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


# select middle 10 degree box for snapshot
lon_mid = (lon_min + lon_max)/2
lat_mid = (lat_min + lat_max)/2

lon_slice_10deg = slice(lon_mid-5,lon_mid+5)
lat_slice_10deg = slice(lat_mid-5,lat_mid+5)

sel_area_10deg = np.where(np.logical_and(np.logical_and(T.XC>lon_mid-5, T.XC<lon_mid+5 ),
                           np.logical_and(T.YC>lat_mid-5, T.YC<lat_mid+5)))

i_min_10deg = min(sel_area_10deg[1])
i_max_10deg = max(sel_area_10deg[1])
j_min_10deg = min(sel_area_10deg[0]) 
j_max_10deg = max(sel_area_10deg[0])


i_slice_10deg = slice(i_min_10deg, i_max_10deg)
j_slice_10deg = slice(j_min_10deg, j_max_10deg)

########################
def save_snapshot(PATH, var, var_save_name):

    # snapshot values
    var_values = var.values

    ########## Save  data
    FULL_PATH_SNAP = PATH+'snapshots/'

    # Check whether path exists or not
    isExist = os.path.exists(FULL_PATH_SNAP)
    if not isExist:
        os.makedirs(FULL_PATH_SNAP)
        print("The new snapshots directory is created!")

    np.save(FULL_PATH_SNAP+var_save_name+'.npy', var_values)
    del var_values

#################
'''
save_snapshot(PATH,T.XC.isel(i=i_slice_10deg,j=j_slice_10deg)[0,:],'lon')
save_snapshot(PATH,T.YC.isel(i=i_slice_10deg,j=j_slice_10deg)[:,0],'lat')

save_snapshot(PATH,B.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'B')
save_snapshot(PATH,U.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'U')
save_snapshot(PATH,V.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'V')
save_snapshot(PATH,W.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'W')
'''
print('MLD')
print(datetime.now())
# sigma0 at 10m depth for reference
sigma0_10m = sigma0.isel(k=6).broadcast_like(sigma0).chunk(chunks={'time': 1, 'k':1, 'j': sigma0.j.size, 'i': sigma0.i.size})

delta_sigma = sigma0 - sigma0_10m
'''
# mixed layer depth
MLD = sigma0.Z.broadcast_like(sigma0).where(delta_sigma<=0.03).min(dim="k",skipna=True).chunk(chunks={'time': 1, 'j': sigma0.j.size, 'i': sigma0.i.size}).rename('Mixed Layer Depth')

save_snapshot(PATH,MLD.isel(time=0,i=i_slice_10deg,j=j_slice_10deg),'MLD')

del MLD
'''
print('FILTER')
print(datetime.now())
# # Filtering: fixed factor scale 24

gcm_filters.required_grid_vars(gcm_filters.GridType.IRREGULAR_WITH_LAND)


# define input parameters

# wet mask
wet_mask = xr.DataArray(np.logical_not(np.isnan(T.Depth)).values,coords={'j':T.j.values , 'i': T.i.values})

# area
area= T.rA

#grid
dxw = xr.DataArray(
    data=U.dxC, 
    coords={'i':T.i,'j':T.j}, 
    dims=('j','i')
)

dyw = xr.DataArray(
    data=U.dyC, 
    coords={'i':T.i,'j':T.j}, 
    dims=('j','i')
)


dxs = xr.DataArray(
    data=U.dxG, 
    coords={'i':T.i,'j':T.j}, 
    dims=('j','i')
)


dys = xr.DataArray(
    data=U.dyG, 
    coords={'i':T.i,'j':T.j}, 
    dims=('j','i')
)

# dx_min
dx_min = min(dxw.min(),dyw.min(),dxs.min(),dys.min())
dx_min = dx_min.values
dx_min

#dx_max
dx_max = max(dxw.max(),dys.max()) #removed ,dyw.max(),dxs.max()
dx_max = dx_max.values
dx_max

#kappa_w and kappa_s
kappa_w_fac = dxw * dxw / (dx_max * dx_max) 
kappa_s_fac = dys * dys / (dx_max * dx_max)



# define filter
filter_factor24 = gcm_filters.Filter(
    filter_scale=24 * dx_max, # factor of 24
    dx_min=dx_min,
    filter_shape=gcm_filters.FilterShape.GAUSSIAN,
    grid_type=gcm_filters.GridType.IRREGULAR_WITH_LAND,
    grid_vars={
        'wet_mask': wet_mask, 
        'dxw': dxw, 'dyw': dyw, 'dxs': dxs, 'dys': dys, 'area': area, 
        'kappa_w': kappa_w_fac, 'kappa_s': kappa_s_fac
    }
)
filter_factor24


# ## Define Mesoscale and submesoscale fields


# mesoscale field defined by the 24 factor filter

Um = filter_factor24.apply(U.where(wet_mask), dims=['i', 'j'])
Vm = filter_factor24.apply(V.where(wet_mask), dims=['i', 'j'])
Wm = filter_factor24.apply(W.where(wet_mask), dims=['i', 'j'])
Bm = filter_factor24.apply(B.where(wet_mask), dims=['i', 'j'])
'''
print('SAVE SNAPSHOT: Bm,Um,Vm,Wm')
print(datetime.now())
save_snapshot(PATH,Bm.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Bm')
save_snapshot(PATH,Um.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Um')
save_snapshot(PATH,Vm.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Vm')
save_snapshot(PATH,Wm.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Wm')
'''
# submesoscale field defined as the deviation between the full resolved field and mesoscale field

Us = U - Um
Vs = V - Vm
Ws = W - Wm
Bs = B - Bm

del U, V, W, B
'''
print('SAVE SNAPSHOT: Bs,Us,Vs,Ws')
print(datetime.now())
save_snapshot(PATH,Bs.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Bs')
save_snapshot(PATH,Us.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Us')
save_snapshot(PATH,Vs.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Vs')
save_snapshot(PATH,Ws.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Ws')
'''
# submesoscale buoyancy fluxes, with a mesoscale filter

UsBs = filter_factor24.apply((Us * Bs).where(wet_mask), dims=['i', 'j'])
VsBs = filter_factor24.apply((Vs * Bs).where(wet_mask), dims=['i', 'j'])
WsBs = filter_factor24.apply((Ws * Bs).where(wet_mask), dims=['i', 'j'])

del Us, Vs, Ws, Bs
'''
print('SAVE SNAPSHOT: UsBs,VsBs,WsBs')
print(datetime.now())
save_snapshot(PATH,UsBs.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'UsBs')
save_snapshot(PATH,VsBs.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'VsBs')
save_snapshot(PATH,WsBs.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'WsBs')

'''
# mesoscale buoyancy gradient
# need to temperarily rechunk for the z derivative
Bm_x = Bm.differentiate("i")
Bm_y = Bm.differentiate("j")
Bm_z = Bm.differentiate("k")
'''
print('SAVE SNAPSHOT: Bm_x,Bm_y,Bm_z')
print(datetime.now())
save_snapshot(PATH,Bm_x.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Bm_x')
save_snapshot(PATH,Bm_y.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Bm_y')
save_snapshot(PATH,Bm_z.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Bm_z')
'''

print('MLD AVERAGE')
print(datetime.now())
# meso velocities and buoyancy MLD average
Um_MLD = Um.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('Um_MLD')
Vm_MLD = Vm.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('Vm_MLD')
Wm_MLD = Wm.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('Wm_MLD')
Bm_MLD = Bm.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('Bm_MLD')
del Um, Vm, Wm, Bm

# submeso fluxes MLD average
UsBs_MLD = UsBs.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('UsBs_MLD')
VsBs_MLD = VsBs.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('VsBs_MLD')
WsBs_MLD = WsBs.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('WsBs_MLD')
del UsBs, VsBs, WsBs

# meso gradients MLD average
Bm_x_MLD = Bm_x.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('Bm_x_MLD')
Bm_y_MLD = Bm_y.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('Bm_y_MLD')
Bm_z_MLD = Bm_z.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('Bm_z_MLD')
del Bm_x, Bm_y, Bm_z



print('COARSE-GRAIN')
print(datetime.now())
### Coarse-grain to 1/4 degree
# building regridder
ds_out_qurter = xr.Dataset(
    {
        "lat": (["lat"], np.arange(lat_min, lat_max, 0.25)),
        "lon": (["lon"], np.arange(lon_min, lon_max, 0.25)),
    }
)

regridder_quarter = xe.Regridder(T.isel(time=0, k=0), ds_out_qurter, "bilinear", periodic=False, reuse_weights=False)
regridder_quarter


# regrid and select middle 10 degree box
lon_mid = (lon_min + lon_max)/2
lat_mid = (lat_min + lat_max)/2

lon_slice_10deg = slice(lon_mid-5,lon_mid+5)
lat_slice_10deg = slice(lat_mid-5,lat_mid+5)

# save 10deg lat lowres and lon lowres grids
lat_lowres = np.arange(lat_min, lat_max, 0.25)
lat_lowres = lat_lowres[np.where(np.logical_and(lat_lowres>=lat_mid-5,lat_lowres<=lat_mid+5))]

lon_lowres = np.arange(lon_min, lon_max, 0.25)
lon_lowres = lon_lowres[np.where(np.logical_and(lon_lowres>=lon_mid-5,lon_lowres<=lon_mid+5))]
'''
np.save(PATH+'snapshots/lat_lowres.npy',lon_lowres)
np.save(PATH+'snapshots/lon_lowres.npy',lat_lowres)
'''

# meso velocities and buoyancy
Um_MLD_lowres = regridder_quarter(Um_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
Vm_MLD_lowres = regridder_quarter(Vm_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
Wm_MLD_lowres = regridder_quarter(Wm_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
Bm_MLD_lowres = regridder_quarter(Bm_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
del Um_MLD, Vm_MLD, Wm_MLD, Bm_MLD

# meso gradients
Bm_x_MLD_lowres = regridder_quarter(Bm_x_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
Bm_y_MLD_lowres = regridder_quarter(Bm_y_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
Bm_z_MLD_lowres = regridder_quarter(Bm_z_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
del Bm_x_MLD, Bm_y_MLD, Bm_z_MLD

# submeso fluxes
UsBs_MLD_lowres = regridder_quarter(UsBs_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg)  
VsBs_MLD_lowres = regridder_quarter(VsBs_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
WsBs_MLD_lowres = regridder_quarter(WsBs_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg)  
del UsBs_MLD, VsBs_MLD, WsBs_MLD

'''
print('SAVE SNAPSHOT LOWRES: Bm,Um,Vm,Wm')
print(datetime.now())
save_snapshot(PATH,Bm_lowres.isel(time=0,k=0),'Bm_lowres')
save_snapshot(PATH,Um_lowres.isel(time=0,k=0),'Um_lowres')
save_snapshot(PATH,Vm_lowres.isel(time=0,k=0),'Vm_lowres')
save_snapshot(PATH,Wm_lowres.isel(time=0,k=0),'Wm_lowres')

print('SAVE SNAPSHOT LOWRES: UsBs,VsBs,WsBs')
print(datetime.now())
save_snapshot(PATH,UsBs_lowres.isel(time=0,k=0),'UsBs_lowres')
save_snapshot(PATH,VsBs_lowres.isel(time=0,k=0),'VsBs_lowres')
save_snapshot(PATH,WsBs_lowres.isel(time=0,k=0),'WsBs_lowres')

print('SAVE SNAPSHOT LOWRES: Bm_x,Bm_y,Bm_z')
print(datetime.now())
save_snapshot(PATH,Bm_x_lowres.isel(time=0,k=0),'Bm_x_lowres')
save_snapshot(PATH,Bm_y_lowres.isel(time=0,k=0),'Bm_y_lowres')
save_snapshot(PATH,Bm_z_lowres.isel(time=0,k=0),'Bm_z_lowres')
'''
# SAVE NETCDF
print('SAVE NETCDF: MLD AVG LOWRES')
print('meso fields')
Um_MLD_lowres.to_netcdf(PATH+'preprocessed_data/Um_MLD_lowres.nc',engine='h5netcdf')
Vm_MLD_lowres.to_netcdf(PATH+'preprocessed_data/Vm_MLD_lowres.nc',engine='h5netcdf')
Wm_MLD_lowres.to_netcdf(PATH+'preprocessed_data/Wm_MLD_lowres.nc',engine='h5netcdf')
Bm_MLD_lowres.to_netcdf(PATH+'preprocessed_data/Bm_MLD_lowres.nc',engine='h5netcdf')

print('submeso fluxes')
UsBs_MLD_lowres.to_netcdf(PATH+'preprocessed_data/UsBs_MLD_lowres.nc',engine='h5netcdf')
VsBs_MLD_lowres.to_netcdf(PATH+'preprocessed_data/VsBs_MLD_lowres.nc',engine='h5netcdf')
WsBs_MLD_lowres.to_netcdf(PATH+'preprocessed_data/WsBs_MLD_lowres.nc',engine='h5netcdf')

print('meso B grad')
Bm_x_MLD_lowres.to_netcdf(PATH+'preprocessed_data/Bm_x_MLD_lowres.nc',engine='h5netcdf')
Bm_y_MLD_lowres.to_netcdf(PATH+'preprocessed_data/Bm_y_MLD_lowres.nc',engine='h5netcdf')
Bm_z_MLD_lowres.to_netcdf(PATH+'preprocessed_data/Bm_z_MLD_lowres.nc',engine='h5netcdf')
'''
print('SAVE SNAPSHOT MLD AVG LOWRES: Bm,Um,Vm,Wm')
print(datetime.now())
save_snapshot(PATH,Bm_lowres_MLD.isel(time=0),'Bm_lowres_MLD')
save_snapshot(PATH,Um_lowres_MLD.isel(time=0),'Um_lowres_MLD')
save_snapshot(PATH,Vm_lowres_MLD.isel(time=0),'Vm_lowres_MLD')
save_snapshot(PATH,Wm_lowres_MLD.isel(time=0),'Wm_lowres_MLD')

print('SAVE SNAPSHOT MLD AVG LOWRES: UsBs,VsBs,WsBs')
print(datetime.now())
save_snapshot(PATH,UsBs_lowres_MLD.isel(time=0),'UsBs_lowres_MLD')
save_snapshot(PATH,VsBs_lowres_MLD.isel(time=0),'VsBs_lowres_MLD')
save_snapshot(PATH,WsBs_lowres_MLD.isel(time=0),'WsBs_lowres_MLD')

print('SAVE SNAPSHOT MLD AVG LOWRES: Bm_x,Bm_y,Bm_z')
print(datetime.now())
save_snapshot(PATH,Bm_x_lowres_MLD.isel(time=0),'Bm_x_lowres_MLD')
save_snapshot(PATH,Bm_y_lowres_MLD.isel(time=0),'Bm_y_lowres_MLD')
save_snapshot(PATH,Bm_z_lowres_MLD.isel(time=0),'Bm_z_lowres_MLD')


print('SAVE DATA: INPUT, OUTPUT, LOSS')
print(datetime.now())
# Save input, output and loss data

def save_data(PATH, dataset_ind, Um_lowres_MLD, Vm_lowres_MLD, Wm_lowres_MLD, Bm_lowres_MLD,UsBs_lowres_MLD, VsBs_lowres_MLD, WsBs_lowres_MLD, Bm_x_lowres_MLD, Bm_y_lowres_MLD, Bm_z_lowres_MLD):
    
    for ii in tqdm(range(len(dataset_ind))):
        # input channels: lowres mesoscale fields averaged over the mixed layer Um, Vm, Wm, Bm
        Um_data = np.nan_to_num(Um_lowres_MLD.Um_lowres_MLD.isel(time=dataset_ind[ii]).values)
        Vm_data = np.nan_to_num(Vm_lowres_MLD.Vm_lowres_MLD.isel(time=dataset_ind[ii]).values)
        Wm_data = np.nan_to_num(Wm_lowres_MLD.Wm_lowres_MLD.isel(time=dataset_ind[ii]).values)
        Bm_data = np.nan_to_num(Bm_lowres_MLD.Bm_lowres_MLD.isel(time=dataset_ind[ii]).values)
        
        # output channels: lowres submeso fluxes averaged over the mixed layer UsBs, VsBs, WsBs
        UsBs_data = np.nan_to_num(UsBs_lowres_MLD.UsBs_lowres_MLD.isel(time=dataset_ind[ii]).values)
        VsBs_data = np.nan_to_num(VsBs_lowres_MLD.VsBs_lowres_MLD.isel(time=dataset_ind[ii]).values)
        WsBs_data = np.nan_to_num(WsBs_lowres_MLD.WsBs_lowres_MLD.isel(time=dataset_ind[ii]).values)
        
        # loss channels: lowres meso buoyancy gradient averaged over the mixed layer Bm_x, Bm_y, Bm_z
        Bm_x_data = np.nan_to_num(Bm_x_lowres_MLD.Bm_x_lowres_MLD.isel(time=dataset_ind[ii]).values)
        Bm_y_data = np.nan_to_num(Bm_y_lowres_MLD.Bm_y_lowres_MLD.isel(time=dataset_ind[ii]).values)
        Bm_z_data = np.nan_to_num(Bm_z_lowres_MLD.Bm_z_lowres_MLD.isel(time=dataset_ind[ii]).values)
        

        # input: 
        cnn_input_ii = np.stack([
            Um_data,
            Vm_data,
            Wm_data,
            Bm_data
        ], axis=0).astype(np.float32)
        
        #output
        cnn_output_ii = np.stack([
            UsBs_data,
            VsBs_data,
            WsBs_data,
        ], axis=0).astype(np.float32)

        #loss
        cnn_loss_ii = np.stack([
            Bm_x_data,
            Bm_y_data,
            Bm_z_data
        ], axis=0).astype(np.float32)
    
    
        ########## Save  data
        FULL_PATH_input = PATH+'input/'
        FULL_PATH_output = PATH+'output/'
        FULL_PATH_loss = PATH+'loss/'

        # Check whether path exists or not
        isExist = os.path.exists(FULL_PATH_input)
        if not isExist:
            os.makedirs(FULL_PATH_input)
            print("The new input directory is created!")
            os.makedirs(FULL_PATH_output)
            print("The new output directory is created!")
            os.makedirs(FULL_PATH_loss)
            print("The new loss directory is created!")


        np.save(FULL_PATH_input+str(ii)+'.npy', cnn_input_ii)
        np.save(FULL_PATH_output+str(ii)+'.npy', cnn_output_ii)
        np.save(FULL_PATH_loss+str(ii)+'.npy', cnn_loss_ii)

    np.save(PATH+'dataset_ind.npy', dataset_ind)


# randomnly generate test and training time indecies

time_ind = len(T.time)
rand_ind = np.arange(time_ind)
np.random.shuffle(rand_ind)
train_percent = 0.7
test_percent = 0.2 
print(f"Dataset: train {np.round(train_percent*100)}%, test {np.round(test_percent*100)}%, val {np.round((1-train_percent-test_percent)*100)}%")
train_ind, test_ind, val_ind =  rand_ind[:round(train_percent*time_ind)], rand_ind[round(train_percent*time_ind):round((train_percent+test_percent)*time_ind)], rand_ind[round((train_percent+test_percent)*time_ind):]                                                                        


# Save train, test and val datasets

# Train set
print('TRAIN: START')
print(datetime.now())
TRAIN_PATH = BASE+CASE_NAME+'train/'
save_data(TRAIN_PATH, train_ind, Um_lowres_MLD, Vm_lowres_MLD, Wm_lowres_MLD, Bm_lowres_MLD,UsBs_lowres_MLD, VsBs_lowres_MLD, WsBs_lowres_MLD, Bm_x_lowres_MLD, Bm_y_lowres_MLD, Bm_z_lowres_MLD)
print('TRAIN: DONE')

# TEST set
print('TEST: START')
print(datetime.now())
TEST_PATH = BASE+CASE_NAME+'test/'
save_data(TEST_PATH, test_ind,  Um_lowres_MLD, Vm_lowres_MLD, Wm_lowres_MLD, Bm_lowres_MLD,UsBs_lowres_MLD, VsBs_lowres_MLD, WsBs_lowres_MLD, Bm_x_lowres_MLD, Bm_y_lowres_MLD, Bm_z_lowres_MLD)
print('TEST: DONE')

# VAL set
print('VAL: START')
print(datetime.now())
VAL_PATH = BASE+CASE_NAME+'val/'
save_data(VAL_PATH, val_ind, Um_lowres_MLD, Vm_lowres_MLD, Wm_lowres_MLD, Bm_lowres_MLD,UsBs_lowres_MLD, VsBs_lowres_MLD, WsBs_lowres_MLD, Bm_x_lowres_MLD, Bm_y_lowres_MLD, Bm_z_lowres_MLD)
print('VAL: DONE')
'''
print('COMPLETE PREPROCESS')
print(datetime.now())





