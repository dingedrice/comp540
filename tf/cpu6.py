# %% [markdown]
# # Imports

# %%
import sys
import tensorflow as tf
import numpy as np
import itertools
import os

from pathlib import Path

import pandas as pd

sys.path.append('gmvae')
from model import GMVAE
from dataset import load_and_mix_data_nolabel


# %% [markdown]
# # Paths

# %%
readdir = 'data/'
savedir = 'results/'
modeldir = 'models/'

# %% [markdown]
# # Load data

# %%
data = np.load(readdir + "d_matrix.npy")
#data_list = []
#for i in range(data.shape[0]):
#   data_list.append(data[i, :, :])
#data = data.reshape(len(data)*10000, 2)
dataset = load_and_mix_data_nolabel(data=data, test_ratio=0.01)


# %% [markdown]
# # Training

# %%
k, n_x, n_z, n_epochs = 5, 531, 2, 4000
qy_dims = [16,16]
qz_dims = [16,16]
pz_dims = [16,16]
px_dims = [16,16]
r_nent = 0.3
batch_size = 2000
lr = 1e-5

model_path = modeldir + 'cpu6model_' +str(k)+'_'+str(n_z)+'_'+str(n_epochs)+'_'+str(r_nent)+'_' + str(batch_size)
results = '_results'

if not os.path.exists(model_path):
    os.makedirs(model_path)

results_dir_s = model_path+results

if not os.path.exists(results_dir_s):
    os.makedirs(results_dir_s)
    
results_dir = Path(results_dir_s)
    
model = GMVAE(model_path, k=k, n_x=n_x, n_z=n_z, qy_dims = qy_dims,
              qz_dims = qz_dims, pz_dims = pz_dims, px_dims = px_dims,
              r_nent = r_nent, batch_size=batch_size, lr=lr)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    history, ye, ze = model.train(dataset, sess, epochs=n_epochs,
                                  n_train_eval=200000, n_test_eval=200000, save_parameters=True, 
                                  is_labeled=False, track_losses=True, verbose=True)
    
f = open(model_path + '/training_params.txt','w+')
f.write(f'k={k}, n_x={n_x}, n_z={n_z}, n_epochs={n_epochs}, qy_dims={qy_dims}, qz_dims={qz_dims}, '+
        f'pz_dims={pz_dims}, px_dims={px_dims}, r_ent={r_nent}, batch_size={batch_size}, lr={lr}')
raw_data = {'k': [k], 'n_x': [n_x], 'n_z': [n_z], 'n_epochs': [n_epochs], 'qy_dims': [qy_dims], 'qz_dims': [qz_dims], 'pz_dims': [pz_dims], 
            'px_dims': [px_dims], 'r_nent': [r_nent], 'batch_size': [batch_size], 'lr': [lr]}
df = pd.DataFrame(data=raw_data)
df.to_pickle(model_path + '/training_params.pkl')
f.close()

# %% [markdown]
# Training curve
