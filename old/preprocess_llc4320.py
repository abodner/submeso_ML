#!/usr/bin/env python
# coding: utf-8

# # Preprocess LLC4320 data

# This program is to preprcess the LLC4320 data using the xmitgcm llcreader package. 
# 
# Note that it needs the 'submeso_env' environment

# In[1]:


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


# In[ ]:


# load full temperature and salinity datasets
ds_T_full = model.get_dataset(varnames=['Theta'], type='latlon')
ds_S_full = model.get_dataset(varnames=['Salt'], type='latlon')


# In[ ]:


# load full velocity dataset 
ds_UV_full = model.get_dataset(varnames=['U','V'], type='latlon')
ds_W_full = model.get_dataset(varnames=['W'], type='latlon')

# interp full velocities onto tracer grid
ds_W_full_interp = ds_W_full.interp(k_l=ds_T_full.k)
ds_UV_full_interp = ds_UV_full.interp(j_g=ds_T_full.j, i_g=ds_T_full.i)


# In[12]:


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
T = ds_T_full.Theta.resample(time='24H').nearest(tolerance="1H").where(sel_area, drop=True)
S = ds_S_full.Salt.resample(time='24H').nearest(tolerance="1H").where(sel_area, drop=True)
U = ds_UV_full_interp.U.resample(time='24H').nearest(tolerance="1H").where(sel_area, drop=True)
V = ds_UV_full_interp.V.resample(time='24H').nearest(tolerance="1H").where(sel_area, drop=True)
W = ds_W_full_interp.W.resample(time='24H').nearest(tolerance="1H").where(sel_area, drop=True)


# In[13]:


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


# In[14]:
# regrid and select middle 10 degree box
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

B.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('B.png')
plt.close()

U.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('U.png')
plt.close()

V.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('V.png')
plt.close()

W.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('W.png')
plt.close()
# In[15]:


# sigma0 at 10m depth for reference
sigma0_10m = sigma0.isel(k=6).broadcast_like(sigma0).chunk(chunks={'time': 1, 'k':1, 'j': sigma0.j.size, 'i': sigma0.i.size})

delta_sigma = sigma0 - sigma0_10m

# mixed layer depth
MLD = sigma0.Z.broadcast_like(sigma0).where(delta_sigma<=0.03).min(dim="k",skipna=True).chunk(chunks={'time': 1, 'j': sigma0.j.size, 'i': sigma0.i.size}).rename('Mixed Layer Depth')


# In[17]:

MLD.isel(time=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('MLD.png')
plt.close()
# # Filtering: fixed factor scale 24

# In[18]:


gcm_filters.required_grid_vars(gcm_filters.GridType.IRREGULAR_WITH_LAND)


# In[19]:


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


# In[20]:


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

# In[21]:


# mesoscale field defined by the 24 factor filter

Um = filter_factor24.apply(U.where(wet_mask), dims=['i', 'j'])
Vm = filter_factor24.apply(V.where(wet_mask), dims=['i', 'j'])
Wm = filter_factor24.apply(W.where(wet_mask), dims=['i', 'j'])
Bm = filter_factor24.apply(B.where(wet_mask), dims=['i', 'j'])


# In[3]:





# In[22]:

Bm.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('Bm.png')
plt.close()

Um.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('Um.png')
plt.close()

Vm.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('Vm.png')
plt.close()

Wm.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('Wm.png')
plt.close()
# In[23]:

# submesoscale field defined as the deviation between the full resolved field and mesoscale field

Us = U - Um
Vs = V - Vm
Ws = W - Wm
Bs = B - Bm


# In[24]:

Bs.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('Bs.png')
plt.close()

Us.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('Us.png')
plt.close()

Vs.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('Vs.png')
plt.close()

Ws.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('Ws.png')
plt.close()
# In[ ]:

# submesoscale buoyancy fluxes

UsBs = Us * Bs
VsBs = Vs * Bs
WsBs = Ws * Bs


# In[ ]:

UsBs.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('UsBs.png')
plt.close()

VsBs.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('VsBs.png')
plt.close()

WsBs.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('WsBs.png')
plt.close()
# In[ ]:

# mesoscale buoyancy gradient
# need to tomperarily rechunk for the z derivative
Bm_x = Bm.differentiate("i")
Bm_y = Bm.differentiate("j")
Bm_z = Bm.chunk(chunks={"time":len(T.time), "k":len(T.k), "j":len(T.j), "i": len(T.i)}).differentiate("k").chunk(chunks={"time":1, "k":1, "j":len(T.j), "i": len(T.i)})

# In[ ]:
Bm_x.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('Bm_x.png')
plt.close()

Bm_y.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('Bm_y.png')
plt.close()

Bm_z.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg).plot()
plt.savefig('Bm_z.png')
plt.close()

# ## Coarse-grain to 1/4 degree
# 
# 

# In[ ]:


# building regridder: coarse grain to 1/4 degree

ds_out_qurter = xr.Dataset(
    {
        "lat": (["lat"], np.arange(lat_min, lat_max, 0.25)),
        "lon": (["lon"], np.arange(lon_min, lon_max, 0.25)),
    }
)

regridder_quarter = xe.Regridder(T.isel(time=0, k=0), ds_out_qurter, "bilinear", periodic=False, reuse_weights=False)
regridder_quarter


# In[ ]:


# regrid and select middle 10 degree box
lon_mid = (lon_min + lon_max)/2
lat_mid = (lat_min + lat_max)/2

lon_slice_10deg = slice(lon_mid-5,lon_mid+5)
lat_slice_10deg = slice(lat_mid-5,lat_mid+5)

# meso velocities and buoyancy
Um_lowres = regridder_quarter(Um).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
Vm_lowres = regridder_quarter(Vm).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
Wm_lowres = regridder_quarter(Wm).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
Bm_lowres = regridder_quarter(Bm).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 

# meso gradients
Bm_x_lowres = regridder_quarter(Bm_x).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
Bm_y_lowres = regridder_quarter(Bm_y).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
Bm_z_lowres = regridder_quarter(Bm_z).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 

# submeso fluxes
UsBs_lowres = regridder_quarter(UsBs).sel(lon=lon_slice_10deg,lat=lat_slice_10deg)  
VsBs_lowres = regridder_quarter(VsBs).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
WsBs_lowres = regridder_quarter(WsBs).sel(lon=lon_slice_10deg,lat=lat_slice_10deg)  


# In[ ]:

Bm_lowres.isel(time=0,k=0).plot()
plt.savefig('Bm_lowres.png')
plt.close()

Um_lowres.isel(time=0,k=0).plot()
plt.savefig('Um_lowres.png')
plt.close()

Vm_lowres.isel(time=0,k=0).plot()
plt.savefig('Vm_lowres.png')
plt.close()

Wm_lowres.isel(time=0,k=0).plot()
plt.savefig('Wm_lowres.png')
plt.close()

Bm_x_lowres.isel(time=0,k=0).plot()
plt.savefig('Bm_x_lowres.png')
plt.close()

Bm_y_lowres.isel(time=0,k=0).plot()
plt.savefig('Bm_y_lowres.png')
plt.close()

Bm_z_lowres.isel(time=0,k=0).plot()
plt.savefig('Bm_z_lowres.png')
plt.close()

UsBs_lowres.isel(time=0,k=0).plot()
plt.savefig('UsBs_lowres.png')
plt.close()

VsBs_lowres.isel(time=0,k=0).plot()
plt.savefig('VsBs_lowres.png')
plt.close()

WsBs_lowres.isel(time=0,k=0).plot()
plt.savefig('WsBs_lowres.png')
plt.close()
# ## Depth averaged

# In[ ]:


# meso velocities and buoyancy
Um_lowres_MLD = Um_lowres.where(delta_sigma<=0.03).min(dim="k",skipna=True).rename('Um_lowres_MLD')
Vm_lowres_MLD = Vm_lowres.where(delta_sigma<=0.03).min(dim="k",skipna=True).rename('Vm_lowres_MLD')
Wm_lowres_MLD = Wm_lowres.where(delta_sigma<=0.03).min(dim="k",skipna=True).rename('Wm_lowres_MLD')
Bm_lowres_MLD = Bm_lowres.where(delta_sigma<=0.03).min(dim="k",skipna=True).rename('Bm_lowres_MLD')

# submeso fluxes 
UsBs_lowres_MLD = UsBs_lowres.where(delta_sigma<=0.03).min(dim="k",skipna=True).rename('UsBs_lowres_MLD')
VsBs_lowres_MLD = VsBs_lowres.where(delta_sigma<=0.03).min(dim="k",skipna=True).rename('VsBs_lowres_MLD')
WsBs_lowres_MLD = WsBs_lowres.where(delta_sigma<=0.03).min(dim="k",skipna=True).rename('WsBs_lowres_MLD')

# meso gradients
Bm_x_lowres_MLD = Bm_x_lowres.where(delta_sigma<=0.03).min(dim="k",skipna=True).rename('Bm_x_lowres_MLD')
Bm_y_lowres_MLD = Bm_y_lowres.where(delta_sigma<=0.03).min(dim="k",skipna=True).rename('Bm_y_lowres_MLD')
Bm_z_lowres_MLD = Bm_z_lowres.where(delta_sigma<=0.03).min(dim="k",skipna=True).rename('Bm_z_lowres_MLD')


# In[ ]:

Bm_lowres_MLD.isel(time=0).plot()
plt.savefig('Bm_lowres_MLD.png')
plt.close()

Um_lowres_MLD.isel(time=0).plot()
plt.savefig('Um_lowres_MLD.png')
plt.close()

Vm_lowres_MLD.isel(time=0).plot()
plt.savefig('Vm_lowres_MLD.png')
plt.close()

Wm_lowres_MLD.isel(time=0).plot()
plt.savefig('Wm_lowres_MLD.png')
plt.close()

Bm_x_lowres_MLD.isel(time=0).plot()
plt.savefig('Bm_x_lowres_MLD.png')
plt.close()

Bm_y_lowres_MLD.isel(time=0).plot()
plt.savefig('Bm_y_lowres_MLD.png')
plt.close()

Bm_z_lowres_MLD.isel(time=0,k=0).plot()
plt.savefig('Bm_z_lowres_MLD.png')
plt.close()

UsBs_lowres_MLD.isel(time=0).plot()
plt.savefig('UsBs_lowres_MLD.png')
plt.close()

VsBs_lowres_MLD.isel(time=0).plot()
plt.savefig('VsBs_lowres_MLD.png')
plt.close()

WsBs_lowres_MLD.isel(time=0).plot()
plt.savefig('WsBs_lowres_MLD.png')
plt.close()
# # Save data: stack input and output, save snapshots

# In[ ]:


def save_data(PATH, CASE_NAME, dataset_ind, Um_lowres_MLD, Vm_lowres_MLD, Wm_lowres_MLD, Bm_lowres_MLD,UsBs_lowres_MLD, VsBs_lowres_MLD, WsBs_lowres_MLD, Bm_x_lowres_MLD, Bm_y_lowres_MLD, Bm_z_lowres_MLD):
    
    for ii in tqdm(range(len(dataset_ind))):
        # input channels: lowres mesoscale fields averaged over the mixed layer Um, Vm, Wm, Bm
        Um_data = np.nan_to_num(Um_lowres_MLD.Um_lowres_MLD.isel(time=dataset_ind[ii]).values)
        Vm_data = np.nan_to_num(Vm_lowres_MLD.Vm_lowres_MLD.isel(time=dataset_ind[ii]).values)
        Wm_data = np.nan_to_num(Wm_lowres_MLD.Wm_lowres_MLD.isel(time=dataset_ind[ii]).values)
        Bm_data = np.nan_to_num(Bm_lowres_MLD.Bm_lowres_MLD.isel(time=dataset_ind[ii]).values)
        
        # output channels: lowres submeso fluxes and meso gradient fields averaged over the mixed layer UsBs, VsBs, WsBs, Bm_x, Bm_y, Bm_z
        UsBs_data = np.nan_to_num(UsBs_lowres_MLD.UsBs_lowres_MLD.isel(time=dataset_ind[ii]).values)
        VsBs_data = np.nan_to_num(VsBs_lowres_MLD.VsBs_lowres_MLD.isel(time=dataset_ind[ii]).values)
        WsBs_data = np.nan_to_num(WsBs_lowres_MLD.WsBs_lowres_MLD.isel(time=dataset_ind[ii]).values)
        
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
            Bm_x_data,
            Bm_y_data,
            Bm_z_data
        ], axis=0).astype(np.float32)


    
    
        ########## Save  data
        FULL_PATH_input = PATH+CASE_NAME+'input/'
        FULL_PATH_output = PATH+CASE_NAME+'output/'

        # Check whether the specified path exists or not
        isExist = os.path.exists(FULL_PATH_input)
        if not isExist:
            os.makedirs(FULL_PATH_input)
            print("The new input directory is created!")
            os.makedirs(FULL_PATH_output)
            print("The new output directory is created!")
        

        np.save(FULL_PATH_input+str(ii)+'.npy', cnn_input_ii)
        np.save(FULL_PATH_output+str(ii)+'.npy', cnn_output_ii)

    np.save(PATH+CASE_NAME+'dataset_ind.npy', dataset_ind)


# In[ ]:


# randomnly generate test and training time indecies (need to make sure no overlapping times)

time_ind = len(T.time)
rand_ind = np.arange(time_ind)
np.random.shuffle(rand_ind)
train_percent = 0.7
test_percent = 0.2 
print(f"Dataset: train {np.round(train_percent*100)}%, test {np.round(test_percent*100)}%, val {np.round((1-train_percent-test_percent)*100)}%")
train_ind, test_ind, val_ind =  rand_ind[:round(train_percent*time_ind)], rand_ind[round(train_percent*time_ind):round((train_percent+test_percent)*time_ind)], rand_ind[round((train_percent+test_percent)*time_ind):]                                                                        


# In[ ]:


# Save train, test and val datasets
CASE_NAME = 'lat_-57_-42_lon_25_40'
BASE = '/scratch/ab10313/submeso_ML_data/'

# Train set
print('TRAIN: START')
TRAIN_PATH = BASE+'train/'
save_data(TRAIN_PATH, CASE_NAME, train_ind, Um_lowres_MLD, Vm_lowres_MLD, Wm_lowres_MLD, Bm_lowres_MLD,UsBs_lowres_MLD, VsBs_lowres_MLD, WsBs_lowres_MLD, Bm_x_lowres_MLD, Bm_y_lowres_MLD, Bm_z_lowres_MLD)
print('DONE')

# TEST set
print('TEST: START')
TEST_PATH = BASE+'test/'
save_data(TEST_PATH, CASE_NAME, test_ind,  Um_lowres_MLD, Vm_lowres_MLD, Wm_lowres_MLD, Bm_lowres_MLD,UsBs_lowres_MLD, VsBs_lowres_MLD, WsBs_lowres_MLD, Bm_x_lowres_MLD, Bm_y_lowres_MLD, Bm_z_lowres_MLD)
print('DONE')

# VAL set
print('VAL: START')
VAL_PATH = BASE+'val/'
save_data(VAL_PATH, CASE_NAME, val_ind, Um_lowres_MLD, Vm_lowres_MLD, Wm_lowres_MLD, Bm_lowres_MLD,UsBs_lowres_MLD, VsBs_lowres_MLD, WsBs_lowres_MLD, Bm_x_lowres_MLD, Bm_y_lowres_MLD, Bm_z_lowres_MLD)
print('DONE')


print('COMPLETE')
