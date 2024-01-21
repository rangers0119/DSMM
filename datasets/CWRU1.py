import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm

# --------------------------------获取数据-----------------------------
# Digital data was collected at 12,000 samples per second
signal_size = 1800
root = 'G:\Chl\DAGCN-main\DAGCN\data\R105'

# HanJie
dataname = {0: ['R105_A_L.mat'],
            1: ['R105_M_O.mat'],
            2: ['R105_P.mat']}
datasetname = ['AL', 'MO', 'P']


axis = ["Y"]

label = [i for i in range(0, 5)]


def get_files(root, N):  # N为转速，N传进来是一个代表负载
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab = []
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            if n == 0:
                path1 = os.path.join(root, datasetname[2], dataname[N[k]][n])
            else:
                path1 = os.path.join(root, datasetname[n - 1], dataname[N[k]][n])
                # path1 = os.path.join(root, datasetname[0], dataname[N[k]][n])
            data1, lab1 = data_load(path1, dataname[N[k]][n], label=label[n])
            data += data1
            lab += lab1

    return [data, lab]


def data_load(filename, axisname, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    datanumber = axisname.split(".")
    # if eval(datanumber[0]) < 100:
    #     realaxis = "X0" + datanumber[0] + axis[0]
    # else:
    #     realaxis = "X" + datanumber[0] + axis[0]
    realaxis = axis[0]
    fl = loadmat(filename)[realaxis]
    fl = fl.flatten()
    fl = fl.reshape(5022000, 1)
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size
    # print(data)
    return data, lab


# --------------------------------------------------------------------------------------------------------------------
class CWRU(object):
    num_classes = 3
    inputchannel = 1

    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
                # Scale(1)
            ])
        }

    def data_split(self, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.3, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.3, random_state=40, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_train, target_val
        else:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.3, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_val


"""
    def data_split(self):

"""
