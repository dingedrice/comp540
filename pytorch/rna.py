# %% [markdown]
# # Rreproduce Training.ipynb (tensorflow implimentation in )

# %%
import torch
import numpy as np

import sys
import itertools
import os

from pathlib import Path

import pandas as pd
sys.path.append('src')
from src.model import GMVAE


readdir = '../data/'
modeldir = 'models/'

data = torch.load(readdir + "d_matrix.pt")

# Hyperparameters
k, n_x, n_z, n_epochs = 5, 531, 2, 401
qy_dims = [16,16]
qz_dims = [16,16]
pz_dims = [16,16]
px_dims = [16,16]
r_nent = 0.3
batch_size = 2000
lr = 1e-5

model_path = modeldir + 'testrna/' 


lengths = [int(p * len(data)) for p in [0.95,0.05]]
tr,v = torch.utils.data.random_split(data,lengths)
train_sampler = torch.utils.data.SubsetRandomSampler(tr.indices)
val_sampler = torch.utils.data.SubsetRandomSampler(v.indices)

# set batch size and set up the data generators for train, val, test sets
trainloader = torch.utils.data.DataLoader(data,batch_size=batch_size,sampler=train_sampler)
valloader = torch.utils.data.DataLoader(data,batch_size=batch_size,sampler=val_sampler)


if not os.path.exists(model_path):
    os.makedirs(model_path)

model1= GMVAE(model_path, k=k, n_x=n_x, n_z=n_z, qy_dims = qy_dims,
              qz_dims = qz_dims, pz_dims = pz_dims, px_dims = px_dims,
              r_nent = r_nent,lr=lr)

model1.train_model(trainloader, valloader, 2001)
