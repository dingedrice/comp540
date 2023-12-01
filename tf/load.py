import sys
import numpy as np
import itertools
import os

from pathlib import Path

import pandas as pd

sys.path.append('gmvae')
from model import GMVAE
from dataset import load_and_mix_data_nolabel

readdir = 'data/'
savedir = 'results/'
modeldir = 'models/'

data = np.load(readdir + "d_matrix.npy")
# data_list = []
# for i in range(data.shape[0]):
#     data_list.append(data[i, :, :])
# data = data.reshape(len(data)*10000, 2)
dataset = load_and_mix_data_nolabel(data=data, test_ratio=0.01)



def write(model_path):
    unpkl = pd.read_pickle(model_path+'/' + 'training_params.pkl')

    k = np.asscalar(unpkl.k[0])
    n_x = np.asscalar(unpkl.n_x[0])
    n_z = np.asscalar(unpkl.n_z[0])
    n_epochs = np.asscalar(unpkl.n_epochs[0])
    qy_dims = (unpkl.qy_dims[0])
    qz_dims = (unpkl.qz_dims[0])
    pz_dims = (unpkl.pz_dims[0])
    px_dims = (unpkl.px_dims[0])
    r_nent = np.asscalar(unpkl.r_nent[0])
    batch_size = np.asscalar(unpkl.batch_size[0])
    lr = np.asscalar(unpkl.lr[0])

    model = GMVAE(model_path, k=k, n_x = n_x, n_z = n_z, qy_dims = qy_dims, #qy_dims = qy_dims,
                qz_dims = qz_dims, pz_dims = pz_dims, px_dims = px_dims, 
                r_nent = r_nent, batch_size = batch_size, lr=lr)

    model.last_epoch = n_epochs

    qy = model.encode_y(data)
    y_pred = np.argmax(qy, axis=1)
    z = model.encode_z(data)

    np.save(model_path+'/y_pred.npy', y_pred)
    np.save(model_path+'/z.npy', z)
    np.save(model_path+'/qy.npy', qy)

#for model_path in [modeldir + 'cpumodel_5_2_2000_0.3_10000', modeldir+'cpumodel_5_2_2000_0.95_10000', modeldir+'cpumodel_5_2_2000_0.95_5000', modeldir + 'cpu4model_5_2_2000_0.3_5000']:
for model_path in [modeldir + 'cpu10model_5_2_3000_0.3_5000', modeldir+'cpu11model_5_2_3000_0.2_5000' ]:
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    write(model_path)
