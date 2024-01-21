import os
import numpy as np
from scipy.io import loadmat
#
root = 'G:\Chl\DAGCN-main\DAGCN\data\R105\AL\R105AL.mat'
root2 = 'G:\Chl\DAGCN-main\DAGCN\data\R105\MO\R105MO.mat'
dataname = {0: ['R105_A_L.mat'],
            1: ['R105_M_O.mat'],
            2: ['R105_P.mat']}
datasetname = ['MO', 'P', 'AL']
#
# path1 = os.path.join(root, datasetname[1], dataname[[0][0]][0])
# fl = loadmat(path1)
#
fl = loadmat(root)['Y']
fl = fl.T
fl = fl.flatten()
fl = fl.reshape(2856960,1)
signal_size1 = 1024
data1 = []
lab =[]
start, end = 0, signal_size1
while end <= fl.shape[0]:
    data1.append(fl[start:end])
    start += signal_size1
    end += signal_size1

fl2 = loadmat(root2)['Y']
fl2 = fl2.T
fl2 = fl2.flatten()
fl2 = fl2.reshape(2856960,1)
signal_size = 1024
data2 = []
start, end = 0, signal_size
while end <= fl.shape[0]:
    data2.append(fl[start:end])
    start += signal_size
    end += signal_size

data = data1+data2