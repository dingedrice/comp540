# comp540

## data/
In data/, The file `convert.ipynb` converted local trajectories into two binary files, `d_matrix.pt` that records the structural information (a sparse distance matrix which is tuned into a 1d array) for RNA molecule for all the frames, and `mg_bound.pkl` that records the coorninate of bound magnesium at each frame. `torch2np.ipynb` converts `d_matrix.pt` into a numpy array file `d_matrix.npy` for tensorflow training. `traj_mb.npy` is from original implimentation and is used for testing.

## pytorch/
pyTorch implimentation of the model. `models/` stores the models, `src/` shows the source code, and `test.ipynb` is for testing with `traj_mb.npy`, comparing with `tf/test_parameter/Training.ipynb`.

## tf/
Tensorflow training on a cluster, so the files are a little messy. `data.ipynb` shows the results.
