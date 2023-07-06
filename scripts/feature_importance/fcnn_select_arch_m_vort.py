#!/usr/bin/env python
# coding: utf-8

# ## Lightning FCNN hyper-parameter sweep 

import wandb
import numpy as np
import sys
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import torch.nn as nn



wandb.login()



BASE = '/scratch/ab10313/pleiades/'
PATH_NN= BASE+'NN_data_smooth/'



import systems.regression_system as regression_system
import models.fcnn as fcnn
import util.metrics as metrics
#import util.misc as misc
#import pyqg_explorer.dataset.forcing_dataset as forcing_dataset



# load preprocessed data into input and output channels

# X INPUT
grad_B = np.load(PATH_NN+'grad_B.npy')
FCOR = np.load(PATH_NN+'FCOR.npy')
Nsquared = np.load(PATH_NN+'Nsquared.npy')
HML = np.load(PATH_NN+'HML.npy')
TAU = np.load(PATH_NN+'TAU.npy')
Q = np.load(PATH_NN+'Q.npy')
HBL = np.load(PATH_NN+'HBL.npy')
div = np.load(PATH_NN+'div.npy')
vort = np.load(PATH_NN+'vort.npy')
#strain = np.load(PATH_NN+'strain.npy')

# note different reshaping of input/output for CNN
X_input = np.stack([FCOR, grad_B, Nsquared, HML, TAU, Q, HBL, div, vort],axis=1)
print('X input shape:')
print( X_input.shape)
print('')


# Y OUTPUT
WB_sg = np.load(PATH_NN+'WB_sg.npy')
WB_sg_mean = np.load(PATH_NN+'WB_sg_mean.npy')
WB_sg_std = np.load(PATH_NN+'WB_sg_std.npy')
              
Y_output = np.tile(WB_sg,(1,1,1,1)).reshape(WB_sg.shape[0],1,WB_sg.shape[1],WB_sg.shape[2]) 
print('Y output shape:')
print(Y_output.shape)
print('')

np.isnan(X_input).any()
np.isnan(Y_output).any()


# TRAIN AND TEST ONLY
# randomnly generate train, test and validation time indecies 
import random
time_ind = X_input.shape[0]
rand_ind = np.arange(time_ind)
rand_seed = 14
random.Random(rand_seed).shuffle(rand_ind)
train_percent = 0.9
test_percent = 0.1 
print(f"Dataset: train {np.round(train_percent*100)}%, test {np.round(test_percent*100)}%")
train_ind, test_ind =  rand_ind[:round(train_percent*time_ind)], rand_ind[round((train_percent)*time_ind):]                                                                        

# check no overlapping indecies
if np.intersect1d(train_ind, test_ind).any():
    print('overlapping indecies')
else:
    print ('no overlapping indecies')
    



# Define X,Y pairs (state, subgrid fluxes) for local network.local_torch_dataset = Data.TensorDataset(
BATCH_SIZE = 64  # Number of sample in each batch


###### training dataset #######
torch_dataset_train = Data.TensorDataset(
    torch.from_numpy(X_input[train_ind]).float() ,
    torch.from_numpy(Y_output[train_ind]).float() ,
)

train_loader = Data.DataLoader(
    dataset=torch_dataset_train, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=10
)
print('TRAIN')
print('X input shape:')
print(torch.from_numpy(X_input[train_ind]).float().shape)
print('Y output shape:')
print(torch.from_numpy(Y_output[train_ind]).float().shape)
print('')

###### test dataset #######
torch_dataset_test = Data.TensorDataset(
    torch.from_numpy(X_input[test_ind]).float(),
    torch.from_numpy(Y_output[test_ind]).float(),    
)

BATCH_SIZE_TEST = BATCH_SIZE#len(torch_dataset_test)

test_loader = Data.DataLoader(
    dataset=torch_dataset_test, 
    batch_size=BATCH_SIZE_TEST, 
    shuffle=False,
    num_workers=10
)

print('TEST')
print('X input shape:')
print(torch.from_numpy(X_input[test_ind]).float().shape)
print('Y output shape:')
print( torch.from_numpy(Y_output[test_ind]).float().shape)
print('')




# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')





seed=123
batch_size=256
input_channels=9
output_channels=1
conv_layers = 3
kernel = 5
kernel_hidden =3
activation="ReLU"
arch="fcnn"
epochs=100
save_path=BASE+"trained_models"
save_name="fcnn_k3_l3_select_m_strain.pt"
lr=0.00024594159283761457
wd=0.023133758465751404

## Wandb config file
config={"seed":seed,
        "lr":lr,
        "wd":wd,
        "batch_size":batch_size,
        "input_channels":input_channels,
        "output_channels":output_channels,
        "activation":activation,
        "save_name":save_name,
        "save_path":save_path,
        "arch":arch,
        "conv_layers":conv_layers,
        "kernel":kernel,
        "kernel_hidden":kernel_hidden,
        "epochs":epochs}



wandb.init(project="submeso_ML",config=config)
model=fcnn.FCNN(config)
config["learnable parameters"]=sum(p.numel() for p in model.parameters())
system=regression_system.RegressionSystem(model,wandb.config["lr"],wandb.config["wd"])
wandb.watch(model, log_freq=1)
wandb_logger = WandbLogger()

trainer = pl.Trainer(
    default_root_dir=model.config["save_path"],
    accelerator="auto",
    max_epochs=config["epochs"],
    enable_progress_bar=False,
    logger=wandb_logger,
    )
trainer.fit(system, train_loader, test_loader)
#model.save_model()
torch.save(model, config["save_path"] + config["save_name"])

#figure_fields=wandb.Image(plot_helpers.plot_fields(pyqg_dataset,model))
#wandb.log({"Fields": figure_fields})

#r2,corr=metrics.get_offline_metrics(model,test_loader)
# figure_power=wandb.Image(figure_power)
# Calculate r2 and corr for test dataset
r2_list = []
corr_list = []
mse_list=[]
for x_data, y_data in test_loader:
    output = model(x_data.to(device))
    mse = nn.MSELoss(y_data.to(device),output)
    r2 = r2_score(output.detach().cpu().numpy(), y_data.detach().cpu().numpy())
    corr, _ = pearsonr(output.detach().cpu().numpy().flatten(), y_data.detach().cpu().numpy().flatten())
    r2_list.append(r2.detach().cpu())
    corr_list.append(corr.detach().cpu())
    mse_list.append(mse.detach().cpu())



# Average r2 and corr values over the test dataset
r2_avg = torch.tensor(r2_list.detach().cpu()).mean().item()
corr_avg = torch.tensor(corr_list.detach().cpu()).mean().item()
test_loss = torch.tensor(mse_list.detach().cpu()).mean().item()

# wandb.log({"Power spectrum": figure_power})
wandb.run.summary["r2"]=r2_avg
wandb.run.summary["corr"]=corr_avg
wandb.run.summary["test_loss"]=test_loss
wandb.finish()
    
    



project_name="submeso_ML"







