import os
import sys
# sys.path.append("/media/hanjie/新加卷/HJ/课题_电流_映射/程序/DAGCN2/DAGCN-main/DAGCN/datasets")
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm
# --------------------------------获取数据-----------------------------
#Digital data was collected at 12,000 samples per second
signal_size = 1024
# root='/media/hanjie/新加卷/HJ/课题_电流_映射/数据/风机电流信号映射处理相同转速/单通道映射/GMMTN新映射'

#电流相同负载不同转速
dataname= {0: ["normal_speed_300_load_0_1.mat", "plane_gear_broken_speed_300_load_0_1.mat", "ring_fault_left_speed_300_load_0_1.mat", "ring_fault_right_speed_300_load_0_1.mat", "sun_gear_broken_speed_300_load_0_1.mat"],  #300rpm
           1: ["normal_speed_480_load_0_1.mat", "plane_gear_broken_speed_480_load_0_1.mat", "ring_fault_left_speed_480_load_0_1.mat", "ring_fault_right_speed_480_load_0_1.mat", "sun_gear_broken_speed_480_load_0_1.mat"],  #480rpm
           2: ["normal_speed_600_load_0_1.mat", "plane_gear_broken_speed_600_load_0_1.mat", "ring_fault_left_speed_600_load_0_1.mat", "ring_fault_right_speed_600_load_0_1.mat", "sun_gear_broken_speed_600_load_0_2.mat"]}  #600rpm

# dataname= {0:["prnormal_speed_300_load_0_1.mat", "prplane_gear_broken_speed_300_load_0_1.mat", "prring_fault_left_speed_300_load_0_1.mat", "prring_fault_right_speed_300_load_0_1.mat", "prsun_gear_broken_speed_300_load_0_1.mat"],  #300rpm
#            1:["prnormal_speed_480_load_0_1.mat", "prplane_gear_broken_speed_480_load_0_1.mat", "prring_fault_left_speed_480_load_0_1.mat", "prring_fault_right_speed_480_load_0_1.mat", "prsun_gear_broken_speed_480_load_0_1.mat"],  #480rpm
#            2:["prnormal_speed_600_load_0_1.mat", "prplane_gear_broken_speed_600_load_0_1.mat", "prring_fault_left_speed_600_load_0_1.mat", "prring_fault_right_speed_600_load_0_1.mat", "prsun_gear_broken_speed_600_load_0_2.mat"]}  #600rpm

datasetname = ["plane_gear_broken", "ring_fault_left", "ring_fault_right", "sun_gear_broken", "Normal"]


# #电流相同转速不同负载
# dataname= {0:["normal_speed_300_load_0_1.mat", "plane_gear_broken_speed_300_load_0_1.mat", "ring_fault_left_speed_300_load_0_1.mat", "ring_fault_right_speed_300_load_0_1.mat", "sun_gear_broken_speed_300_load_0_11.mat"],
#           1:["normal_speed_300_load_1_1.mat", "plane_gear_broken_speed_300_load_2_1.mat", "ring_fault_left_speed_300_load_2_1.mat", "ring_fault_right_speed_300_load_2_1.mat", "sun_gear_broken_speed_300_load_2_11.mat"],
#           2:["normal_speed_300_load_2_1.mat", "plane_gear_broken_speed_300_load_4_1.mat", "ring_fault_left_speed_300_load_4_1.mat", "ring_fault_right_speed_300_load_4_1.mat", "sun_gear_broken_speed_300_load_4_11.mat"]}


# dataname= {0:["prnormal_speed_300_load_0_1.mat", "prplane_gear_broken_speed_300_load_0_1.mat", "prring_fault_left_speed_300_load_0_1.mat", "prring_fault_right_speed_300_load_0_1.mat", "prsun_gear_broken_speed_300_load_0_11.mat"],
#           1:["prnormal_speed_300_load_1_1.mat", "prplane_gear_broken_speed_300_load_2_1.mat", "prring_fault_left_speed_300_load_2_1.mat", "prring_fault_right_speed_300_load_2_1.mat", "prsun_gear_broken_speed_300_load_2_11.mat"],
#           2:["prnormal_speed_300_load_2_1.mat", "prplane_gear_broken_speed_300_load_4_1.mat", "prring_fault_left_speed_300_load_4_1.mat", "prring_fault_right_speed_300_load_4_1.mat", "prsun_gear_broken_speed_300_load_4_11.mat"]}
#

############德国数据集实验一

# dataname= {0:["N15_M01_F10_K001_1.mat", "N15_M01_F10_KA03_1.mat", "N15_M01_F10_KA05_1.mat", "N15_M01_F10_KI03_1.mat", "N15_M01_F10_KI07_1.mat"],
#           1:["N15_M07_F04_K001_1.mat", "N15_M07_F04_KA03_1.mat", "N15_M07_F04_KA05_1.mat", "N15_M07_F04_KI03_1.mat", "N15_M07_F04_KI07_1.mat"],
#           2:["N15_M07_F10_K001_1.mat", "N15_M07_F10_KA03_1.mat", "N15_M07_F10_KA05_1.mat", "N15_M07_F10_KI03_1.mat", "N15_M07_F10_KI07_1.mat"]}

# dataname= {0:["prN15_M01_F10_K001_1.mat", "prN15_M01_F10_KA03_1.mat", "prN15_M01_F10_KA05_1.mat", "prN15_M01_F10_KI03_1.mat", "prN15_M01_F10_KI07_1.mat"],
#           1:["prN15_M07_F04_K001_1.mat", "prN15_M07_F04_KA03_1.mat", "prN15_M07_F04_KA05_1.mat", "prN15_M07_F04_KI03_1.mat", "prN15_M07_F04_KI07_1.mat"],
#           2:["prN15_M07_F10_K001_1.mat", "prN15_M07_F10_KA03_1.mat", "prN15_M07_F10_KA05_1.mat", "prN15_M07_F10_KI03_1.mat", "prN15_M07_F10_KI07_1.mat"]}
#
#
# datasetname = ["KA03", "KA05", "KI03", "KI07", "K001"]



# 映射德国2
# dataname= {0:["prN15_M01_F10_K001_1.mat", "prN15_M01_F10_KA01_1.mat", "prN15_M01_F10_KA03_1.mat", "prN15_M01_F10_KI01_1.mat", "prN15_M01_F10_KI07_1.mat"],
#           1:["prN15_M07_F04_K001_1.mat", "prN15_M07_F04_KA01_1.mat", "prN15_M07_F04_KA03_1.mat", "prN15_M07_F04_KI01_1.mat", "prN15_M07_F04_KI07_1.mat"],
#           2:["prN15_M07_F10_K001_1.mat", "prN15_M07_F10_KA01_1.mat", "prN15_M07_F10_KA03_1.mat", "prN15_M07_F10_KI01_1.mat", "prN15_M07_F10_KI07_1.mat"]}
#
#
# datasetname = [ "KA01", "KA03", "KI01", "KI07","K001"]

# ##德国数据集实验二
#
# dataname= {0:["N15_M01_F10_K001_1.mat", "N15_M01_F10_KA01_1.mat", "N15_M01_F10_KA03_1.mat", "N15_M01_F10_KI01_1.mat", "N15_M01_F10_KI07_1.mat"],
#           1:["N15_M07_F04_K001_1.mat", "N15_M07_F04_KA01_1.mat", "N15_M07_F04_KA03_1.mat", "N15_M07_F04_KI01_1.mat", "N15_M07_F04_KI07_1.mat"],
#           2:["N15_M07_F10_K001_1.mat", "N15_M07_F10_KA01_1.mat", "N15_M07_F10_KA03_1.mat", "N15_M07_F10_KI01_1.mat", "N15_M07_F10_KI07_1.mat"]}
#
# #
# datasetname = [ "KA01", "KA03", "KI01", "KI07","K001"]


####华工
# dataname= {0:["1500_BrokenTeeth_load0","1500_Empty_load0", "1500_normal_load0", "1500_pitting_load0"],
#            1:["1500_BrokenTeeth_load40","1500_Empty_load40", "1500_normal_load40", "1500_pitting_load40"],
#            2:["1500_BrokenTeeth_load60","1500_Empty_load60", "1500_normal_load60", "1500_pitting_load60"]}

# datasetname = ["empty", "normal", "pitting",
#                "broken"]
#
# dataname= {0:["1500rpm_BrokenTeeth","1500rpm_Empty", "1500rpm_normal", "1500rpm_pitting"],
#            1:["1800rpm_BrokenTeeth","1800rpm_Empty", "1800rpm_normal", "1800rpm_pitting"],
#            2:["2400rpm_BrokenTeeth","2400rpm_Empty", "2400rpm_normal", "2400rpm_pitting"]}


###振动相同负载
# dataname= {0:["normal_speed_300_load_0.mat", "plane_gear_broken_speed_300_load_0.mat", "plane_gear_crack_1mm_speed_300_load_0.mat", "ring_fault_left_speed_300_load_0.mat", "sun_gear_broken_speed_300_load_0.mat"],  #300rpm
#            1:["normal_speed_480_load_0.mat", "plane_gear_broken_speed_480_load_0.mat", "plane_gear_crack_1mm_speed_480_load_0.mat", "ring_fault_left_speed_480_load_0.mat", "sun_gear_broken_speed_480_load_0.mat"],  #480rpm
#            2:["normal_speed_600_load_0.mat", "plane_gear_broken_speed_600_load_0.mat", "plane_gear_crack_1mm_speed_600_load_0.mat", "ring_fault_left_speed_600_load_0.mat", "sun_gear_broken_speed_600_load_0.mat"],
#            3:["normal_speed_720_load_0.mat", "plane_gear_broken_speed_720_load_0.mat", "plane_gear_crack_1mm_speed_720_load_0.mat", "ring_fault_left_speed_720_load_0.mat", "sun_gear_broken_speed_720_load_0.mat"]}  #600rpm

####振动相同转速
# dataname= {0:["normal_speed_300_load_0_1","plane_gear_broken_speed_300_load_0_1", "plane_gear_crack_1mm_speed_300_load_0_1", "ring_fault_left_speed_300_load_0_1", "sun_gear_broken_speed_300_load_0_1"],
#            1:["normal_speed_300_load_1_1","plane_gear_broken_speed_300_load_2_1", "plane_gear_crack_1mm_speed_300_load_2_1", "ring_fault_left_speed_300_load_0_1", "sun_gear_broken_speed_300_load_2_1"],
#            2:["normal_speed_300_load_2_1","plane_gear_broken_speed_300_load_4_1", "plane_gear_crack_1mm_speed_300_load_4_1", "ring_fault_left_speed_300_load_0_1", "sun_gear_broken_speed_300_load_4_1"]}
#
# datasetname = ["plane_gear_broken", "plane_gear_crack_1mm", "ring_fault_left","sun_gear_broken",
#                "Normal"]


######靶车数据
# dataname = {0:["prE_ds12_g_fault1_speed100_1","prE_ds12_g_fault2_speed100_1","prE_ds12_g_fault3_speed100_1","prE_ds12_g_fault4_speed100_1", "prE_ds12_g_normal_speed100_1"],
#             1:["prE_ds12_g_fault1_speed300_1","prE_ds12_g_fault2_speed300_1","prE_ds12_g_fault3_speed300_1","prE_ds12_g_fault4_speed300_1", "prE_ds12_g_normal_speed300_1"],
#             2:["prE_ds12_g_fault1_speed500_1","prE_ds12_g_fault2_speed500_1","prE_ds12_g_fault3_speed500_1","prE_ds12_g_fault4_speed500_1", "prE_ds12_g_normal_speed500_1"]}

# dataname = {0:["E_ds12_g_fault1_speed100_1","E_ds12_g_fault2_speed100_1","E_ds12_g_fault3_speed100_1","E_ds12_g_fault4_speed100_1", "E_ds12_g_normal_speed100_1"],
#             1:["E_ds12_g_fault1_speed300_1","E_ds12_g_fault2_speed300_1","E_ds12_g_fault3_speed300_1","E_ds12_g_fault4_speed300_1", "E_ds12_g_normal_speed300_1"],
#             2:["E_ds12_g_fault1_speed500_1","E_ds12_g_fault2_speed500_1","E_ds12_g_fault3_speed500_1","E_ds12_g_fault4_speed500_1", "E_ds12_g_normal_speed500_1"]}
#
#dataname= {0:["N15_M01_F10_K001_1.mat", "N15_M01_F10_KA03_1.mat", "N15_M01_F10_KA05_1.mat", "N15_M01_F10_KI03_1.mat", "N15_M01_F10_KI07_1.mat"],
          1:["N15_M07_F04_K001_1.mat", "N15_M07_F04_KA03_1.mat", "N15_M07_F04_KA05_1.mat", "N15_M07_F04_KI03_1.mat", "N15_M07_F04_KI07_1.mat"],
          2:["N15_M07_F10_K001_1.mat", "N15_M07_F10_KA03_1.mat", "N15_M07_F10_KA05_1.mat", "N15_M07_F10_KI03_1.mat", "N15_M07_F10_KI07_1.mat"]}

#
# datasetname = ["fault2", "fault3", "fault4","normal", "fault1"]


#######机械手
# dataname= {0:["E_g_fault1_load0_speed120_1.mat", "E_g_fault2_load0_speed120_1.mat", "E_g_fault3_load0_speed120_1.mat", "E_g_fault4_load0_speed120_1.mat", "E_g_normal_load0_speed120_1.mat"],
#           1:["E_g_fault1_load5_speed120_1.mat",  "E_g_fault2_load5_speed120_1.mat", "E_g_fault3_load5_speed120_1.mat", "E_g_fault4_load5_speed120_1.mat", "E_g_normal_load5_speed120_1.mat"],
#           2:["E_g_fault1_load10_speed120_1.mat", "E_g_fault2_load10_speed120_1.mat","E_g_fault3_load10_speed120_1.mat", "E_g_fault4_load10_speed120_1.mat", "E_g_normal_load10_speed120_1.mat"]}
#
# datasetname = [ "fault2", "fault3", "fault4", "normal","fault1"]



######机械手映射
# dataname= {0:["prE_g_fault1_load0_speed120_1.mat", "prE_g_fault2_load0_speed120_1.mat", "prE_g_fault3_load0_speed120_1.mat", "prE_g_fault4_load0_speed120_1.mat", "prE_g_normal_load0_speed120_1.mat"],
#           1:["prE_g_fault1_load5_speed120_1.mat",  "prE_g_fault2_load5_speed120_1.mat", "prE_g_fault3_load5_speed120_1.mat", "prE_g_fault4_load5_speed120_1.mat", "prE_g_normal_load5_speed120_1.mat"],
#           2:["prE_g_fault1_load10_speed120_1.mat", "prE_g_fault2_load10_speed120_1.mat","prE_g_fault3_load10_speed120_1.mat", "prE_g_fault4_load10_speed120_1.mat", "prE_g_normal_load10_speed120_1.mat"]}
#
# datasetname = [ "fault2", "fault3", "fault4", "normal","fault1"]



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
                path1 = os.path.join(root, datasetname[4], dataname[N[k]][n])
            else:
                path1 = os.path.join(root, datasetname[n - 1], dataname[N[k]][n])
                # path1 = os.path.join(root, datasetname[0], dataname[N[k]][n])
            data1, lab1 = data_load(path1, label=label[n])
            data += data1
            lab += lab1

    return [data, lab]


def data_load(filename, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''


    ####风机电流信号输入
    a = loadmat(filename)['Signal_0']['y_values']
    b = a[0, 0]['values']
    c = b[0, 0]
    fl = c[0:51200, 1]
    fl = fl.T
    fl = fl.flatten()
    fl = fl.reshape(51200, 1)
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab


    # # 映射
    # a = loadmat(filename)['Y']
    # fl = a.T
    # fl = fl.flatten()
    # fl = fl[0:266240]
    # fl = fl.reshape(266240, 1)
    # data = []
    # lab = []
    # start, end = 0, signal_size
    # while end <= fl.shape[0]:
    #     data.append(fl[start:end])
    #     lab.append(label)
    #     start += signal_size
    #     end += signal_size
    #
    # # print(data)
    # return data, lab



    # #####德国数据输入
    # a = loadmat(filename)['c']
    # fl = a[0, 0:256000]
    # fl = fl.T
    # fl = fl.flatten()
    # fl = fl.reshape(256000, 1)
    # data = []
    # lab = []
    # start, end = 0, signal_size
    # while end <= fl.shape[0]:
    #     data.append(fl[start:end])
    #     lab.append(label)
    #     start += signal_size
    #     end += signal_size
    #
    # return data, lab

    # 映射数据德国
    # a = loadmat(filename)['Y']
    # fl = a[0, 0:256000]
    # fl = fl.T
    # fl = fl.flatten()
    # fl = fl.reshape(256000, 1)
    # data = []
    # lab = []
    # start, end = 0, signal_size
    # while end <= fl.shape[0]:
    #     data.append(fl[start:end])
    #     lab.append(label)
    #     start += signal_size
    #     end += signal_size
    #
    # return data, lab



    ####华工
    # a = loadmat(filename)['Y']
    # fl = a
    # fl = fl.T
    # fl = fl.flatten()
    # fl = fl.reshape(102400, 1)
    # data = []
    # lab = []
    # start, end = 0, signal_size
    # while end <= fl.shape[0]:
    #     data.append(fl[start:end])
    #     lab.append(label)
    #     start += signal_size
    #     end += signal_size
    #
    # return data, lab

    ####风机振动信号输入相同转速
    # a = loadmat(filename)['Signal_1']['y_values']
    # b = a[0, 0]['values']
    # c = b[0, 0]
    # fl = c[0:51200, 0]
    # fl = fl.T
    # fl = fl.flatten()
    # fl = fl.reshape(51200, 1)
    # data = []
    # lab = []
    # start, end = 0, signal_size
    # while end <= fl.shape[0]:
    #     data.append(fl[start:end])
    #     lab.append(label)
    #     start += signal_size
    #     end += signal_size
    #
    # return data, lab

####风机振动信号输入相同负载
    # a = loadmat(filename)['Y']
    # fl = a[0:51200, 0]
    # fl = fl.T
    # fl = fl.flatten()
    # fl = fl.reshape(51200, 1)
    # data = []
    # lab = []
    # start, end = 0, signal_size
    # while end <= fl.shape[0]:
    #     data.append(fl[start:end])
    #     lab.append(label)
    #     start += signal_size
    #     end += signal_size
    #
    # return data, lab



###############################################机械手数据输入
    # a = loadmat(filename)['Y']
    # fl = a[0, 0:143360]
    # fl = fl.T
    # fl = fl.flatten()
    # fl = fl.reshape(143360, 1)
    # data = []
    # lab = []
    # start, end = 0, signal_size
    # while end <= fl.shape[0]:
    #     data.append(fl[start:end])
    #     lab.append(label)
    #     start += signal_size
    #     end += signal_size
    #
    # return data, lab

#--------------------------------------------------------------------------------------------------------------------
class CWRU(object):
    num_classes = 5
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
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_train, target_val
        else:
            #get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
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
if __name__ == '__main__':
    a = loadmat('/media/hanjie/新加卷/HJ/课题_电流_映射/数据/风机电流信号映射处理相同转速/单通道映射/GMMTN新映射/0-0/Normal/prnormal_speed_300_load_0_1.mat')['Y']
    fl = a.T
    fl = fl.flatten()
    fl = fl.reshape(51200, 1)
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size

    print(len(data))
    print(data[0].shape)
    print(label)








