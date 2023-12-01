# comp540

## data/
In data/, The file `convert.ipynb` converted local trajectories into two binary files, `d_matrix.pt` that records the structural information (a sparse distance matrix which is tuned into a 1d array) for RNA molecule for all the frames, that is the input vector for our training. And `mg_bound.pkl` that records the coorninate of bound magnesium at each frame, which we do not have time analyze, unfortunately. `torch2np.ipynb` converts `d_matrix.pt` into a numpy array file `d_matrix.npy` for tensorflow training. `traj_mb.npy` is from original implimentation and is used for testing.

## pytorch/
pyTorch implimentation of the model. `models/` stores the loss and parameters of models, `src/` shows the source code. `test.ipynb` is for testing purpose using data from `traj_mb.npy` and making comparisons with `tf/test_parameter/Training.ipynb`. `rna.ipynb` is for training with `d_matrix.pt`, which showed severe minimization issue, forcing us going back to Tensorflow (the data in folder `testrna/` were otained from runs on supercomputer using `rna.py`, and `rna.ipynb` loaded the final results instead of training it again).

## tf/
Tensorflow training were all finished on a supercomputer cluster, and each folder contains a different model with diffeent hyperparameters. The source code for the training can be found in folder `gmvae/` or the [original Github repo](https://github.com/yabozkurt/gmvae). The `slurm*.out` files in each folder indicates the loss and entropy along training and can be converted to a readble file for numpy using `grep_loss.sh`. Reconstruction file `*npy` in each folder were created with `load.py`. The folder `original_5_1_401_0.05_5000` contains training results with `traj_mb.npy`, which makes the second column of Table.1 in the final report. The parameters were stored in files from `1.npy` to `22.npy` and were tested in `pytorch/test.ipynb` for ppyTorch models. `data.ipynb` reads the reconstruction for each model and plotted the density and clustering diagrams and all other figures for analysis. `cluster.gro` contains the average structure information. You may need package `mdtraj` to rerun the code amd **vmd** or **pymol** software to visualize the structure.
