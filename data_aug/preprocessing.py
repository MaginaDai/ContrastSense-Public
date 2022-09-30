from cProfile import label
from gettext import find
from itertools import count
import os
from os.path import dirname
import pdb
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from scipy.interpolate import interp1d
from collections import Counter

sys.path.append(dirname(sys.path[0]))
sys.path.append(dirname(dirname(sys.path[0])))
from data_aug.HHAR import preprocess_hhar
from data_aug.LPF import filter_dataset

movement = ['stand', 'sit', 'walk', 'stairsup', 'stairsdown', 'bike']
devices = ['nexus4_1', 'nexus4_2', 's3_1', 's3_2', 's3mini_1', 's3mini_2', 'samsungold_1', 'samsungold_2']
models = ['nexus4', 's3', 's3mini', 'samsungold']
users = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
uot_movement = ['Standing', 'Sitting', 'Walking', 'Upstairs', 'Downstairs',  'Running']

MAX_INDEX = {
    'HHAR': 9166,
    'MotionSense': 4530,
    'Shoaib': 10334,
    'UCI': 2087,
    'ICHAR': 9152,
    'HASC': 10291,
}

UsersPosition = {
    'HHAR': -3,
    'MotionSense': 0,
    'Shoaib': -1,
    'UCI': -1,
    'ICHAR': -2,
    'HASC': 0,
}

LabelPosition = {
    'HHAR': -1,
    'MotionSense': -1,
    'Shoaib': -2,
    'UCI': -2,
    'ICHAR': -1,
    'HASC': -1
}

ClassesNum = {
    'HHAR': 6,
    'MotionSense': 6,
    'Shoaib': 7,
    'UCI': 6,
    'ICHAR': 9,
    'HASC': 6
}


samplingRate = 25
window = 100
train_ratio = 0.8
test_ratio = 0.2
tune_ratio = 0.2
test_num_of_user = 3


# MAX_INDEX = 9166
percent = [0.2, 0.5, 1, 2, 5, 10]


def preprocessing_HHAR_cross_person(main_dir):
    num = MAX_INDEX
    u = []
    for i in range(MAX_INDEX):
        sub_dir = main_dir + str(i) + '.npz'
        data = np.load(sub_dir, allow_pickle=True)
        u.append(data['add_infor'][0, -3])

    users_test_idx = np.random.randint(0, len(users), 2)
    users_test_name = [users[i] for i in users_test_idx]
    test_num = [j for j in range(num) if u[j] in users_test_name]
    non_test_num = [j for j in range(num) if u[j] not in users_test_name]

    train_set_len = int(len(non_test_num) * 0.9)
    train_num, val_num = random_split(non_test_num, [train_set_len, len(non_test_num) - train_set_len])
    write_dataset(dir, train_num, val_num, test_num)
    return


def preprocessing_dataset_cross_person(dir, dataset):
    print(dataset)

    num = MAX_INDEX[dataset]
    u = []
    for i in range(num):
        sub_dir = dir + str(i) + '.npz'
        data = np.load(sub_dir, allow_pickle=True)
        u.append(data['add_infor'][0, UsersPosition[dataset]])
    user_type = np.unique(u)
    
    print(user_type)
    users_test_idx = np.random.randint(0, len(user_type), int(len(user_type) * 0.25))
    users_test_name = [user_type[i] for i in users_test_idx]
    print(users_test_name)
    test_num = [j for j in range(num) if u[j] in users_test_name]
    non_test_num = [j for j in range(num) if u[j] not in users_test_name]

    train_set_len = int(len(non_test_num) * 0.9)
    train_num, val_num = random_split(non_test_num, [train_set_len, len(non_test_num) - train_set_len])

    write_dataset(dir, train_num, val_num, test_num)
    write_tune_set(dir)
    # write_balance_tune_set(dir, dataset)
    return


def preprocessing_dataset_cross_person_val(dir, dataset, test_portion=0.6, val_portion=0.15):
    print(dataset)
    
    file_name_list = [file for file in os.listdir(dir) if 'set' not in file]
    
    num = len(file_name_list)
    u = []
    # label_distribution = np.zeros(6)
    for i in range(num):
        sub_dir = dir + str(i) + '.npz'
        data = np.load(sub_dir, allow_pickle=True)
        u.append(data['add_infor'][0, UsersPosition[dataset]])
        # label_distribution[int(data['add_infor'][0, -1])] += 1
    
    # print(label_distribution)
    user_type = np.unique(u)
    test_num = int(len(user_type) * test_portion)
    val_num = int(len(user_type) * val_portion)
    
    print(len(user_type))
    np.random.shuffle(user_type)
    users_test_name = np.sort(user_type[:test_num])
    # users_test_name = np.array(['e', 'i'])
    print(len(users_test_name))

    users_train_name = np.sort(user_type[test_num+val_num:])
    # users_train_name = np.array(['a', 'd', 'f', 'g', 'h'])
    print(len(users_train_name))
    
    users_val_name = np.sort(user_type[test_num:test_num+val_num])
    # users_val_name = np.array(['b', 'c'])
    print(len(users_val_name))

    test_num = [j for j in range(num) if u[j] in users_test_name]
    train_num = [j for j in range(num) if u[j] in users_train_name]
    val_num =  [j for j in range(num) if u[j] in users_val_name]

    # write_dataset(dir, train_num, val_num, test_num)
    # write_balance_tune_set(dir, dataset, dataset_size=num)
    return


def write_dataset(dir, train_num, val_num, test_num):
    train_num = list(train_num)
    train_num.sort()
    val_num = list(val_num)
    val_num.sort()
    test_num = list(test_num)
    test_num.sort()

    train_set = []
    val_set = []
    test_set = []

    for n in train_num:
        train_set.append(str(n) + '.npz')
    for n in val_num:
        val_set.append(str(n) + '.npz')
    for n in test_num:
        test_set.append(str(n) + '.npz')

    train_set = np.asarray(train_set)
    val_set = np.asarray(val_set)
    test_set = np.asarray(test_set)

    print(len(train_set))
    print(len(val_set))
    print(len(test_set))
    np.savez(os.path.join(dir, 'train_set' + '.npz'), train_set=train_set)
    np.savez(os.path.join(dir, 'val_set' + '.npz'), val_set=val_set)
    np.savez(os.path.join(dir, 'test_set' + '.npz'), test_set=test_set)    
    return

def write_tune_set(dir):
    loc = dir + 'train_set' + '.npz'
    data = np.load(loc)
    train_set = data['train_set']
    whole_set = train_set
    whole_set.sort()
    np.random.shuffle(whole_set)
    set_size = len(whole_set)
    for per in percent:
        tune_set = whole_set[:int(per*0.01*set_size)]
        tune_set.sort()
        print(len(tune_set))
        loc = dir + 'tune_set_' + str(per).replace('.', '_') + '.npz'
        np.savez(loc, tune_set=tune_set)
    return


def write_balance_tune_set(dir, dataset, dataset_size=None):
    loc = dir + 'train_set' + '.npz'
    data = np.load(loc)
    train_set = data['train_set']

    label = []

    for i in train_set:
        sub_dir = dir + i
        data = np.load(sub_dir, allow_pickle=True)
        label.append(data['add_infor'][0, LabelPosition[dataset]])

    label = np.array(label)
    label_type = np.unique(label)
    label_type_num = len(label_type)

    print(label_type)
    assert label_type_num == ClassesNum[dataset]

    print(f"motion classes {label_type_num}, total train num {len(label)}")

    if dataset_size is None:
        set_size = MAX_INDEX[dataset]  # select 1% of the whole dataset from training set.
    else:
        set_size = dataset_size
    for per in percent:
        label_num = per*0.01*set_size
        label_per_class = int(label_num / label_type_num)
        tune_set = []
        counter = []
        if label_per_class < 1:
            label_per_class = 1  # at least one label for each class
            print("at least one sample per class")
        
        for i in label_type:
            idx = np.argwhere(label == i).squeeze()
            np.random.shuffle(idx)
            tune_set.extend(train_set[idx[:label_per_class]])
            counter.append(len(list(idx[:label_per_class])))

        print(f"percent {per}: {len(tune_set)}")
        print(f"each class {counter}")
        loca = dir + 'tune_set_' + str(per).replace('.', '_') + '.npz'
        np.savez(loca, tune_set=tune_set)
    return


def datasets_shot_record(dir, datasets, set_type='tune_set'):

    tune_dir = dir + set_type + '_1.npz'
    data = np.load(tune_dir, allow_pickle=True)
    data = data[set_type]
    label_type = []
    label = []
    print(len(data))
    for i in data:
        sub_dir = dir + str(i)
        d = np.load(sub_dir, allow_pickle=True)
        label.append(d['add_infor'][0, LabelPosition[datasets]])

    label_type = np.unique(label)
    counter = np.zeros(len(label_type))
    for i in label:
        counter[np.where(label_type==i)]+=1

    print(f'{datasets}:  {counter} in total {sum(counter)}')

    return


def datasets_users_record(dir, datasets):
    file_name_list = [file for file in os.listdir(dir) if 'set' not in file]
    # label_distribution = np.zeros(6)
    users=[]
    for i in file_name_list:
        d = np.load(dir+i, allow_pickle=True)
        users.append(d['add_infor'][0, UsersPosition[datasets]])
    
    users_list = np.unique(np.array(users))

    # print(users_list)
    print(f'{len(users_list)}, 0.25: {int(len(users_list) *0.25)}, 0.15:{int(len(users_list) *0.15)}, 0.6:{int(len(users_list) *0.6)}')


def seg_different_test_num():
    path = r'./datasets/HHAR_50_200'
    for t in range(2, 7):
        path_save = path + r'_test_{}/'.format(t)
        preprocessing_dataset_cross_person_val(dir=path_save, dataset='HHAR', dataset_size=13047, test_num=t)
    return


def extract_and_seg_hhar(path_save, dataset, window_time, seq_len, version, test_num):
    num = preprocess_hhar(DATASET_PATH, path_save, version=version, window_time=window_time, seq_len=seq_len)  # use jump to control overlap.
    preprocessing_dataset_cross_person_val(dir=path_save, dataset=dataset, dataset_size=num, test_num=test_num)
    return

DATASET_PATH = r'./original_dataset/hhar/'

if __name__ == '__main__':
    # divide_fewer_labels()
    # data = np.load(os.path.join(path_save, 'val_set' + '.npz'))
    # val_set = data['train_set']
    # np.savez(os.path.join(path_save, 'val_set' + '.npz'), val_set=val_set)
    #
    # data = np.load(os.path.join(path_save, 'test_set' + '.npz'))
    # test_set = data['train_set']
    # np.savez(os.path.join(path_save, 'test_set' + '.npz'), test_set=test_set)

    # preprocessing_HHAR_cross_person(main_dir=r'../datasets/HHAR person/')

    preprocessing_dataset_cross_person_val(dir=r'datasets/HHAR_50_200/', dataset='HHAR')
    preprocessing_dataset_cross_person_val(dir=r'datasets/MotionSense_50_200/', dataset='MotionSense')
    preprocessing_dataset_cross_person_val(dir=r'datasets/Shoaib_50_200/', dataset='Shoaib')
    preprocessing_dataset_cross_person_val(dir=r'datasets/UCI_50_200/', dataset='UCI')
    preprocessing_dataset_cross_person_val(dir=r'datasets/ICHAR_50_200/', dataset='ICHAR')
    preprocessing_dataset_cross_person_val(dir=r'datasets/HASC_50_200/', dataset='HASC')
 
    # datasets_shot_record(dir=r'datasets/HHAR_50_200/', datasets='HHAR')

    # datasets_users_record(dir=r'datasets/HHAR_50_200/', datasets='HHAR')
    # datasets_users_record(dir=r'datasets/MotionSense_50_200/', datasets='MotionSense')
    # datasets_users_record(dir=r'datasets/Shoaib_50_200/', datasets='Shoaib')
    # datasets_users_record(dir=r'datasets/UCI_50_200/', datasets='UCI')
    # datasets_users_record(dir=r'datasets/ICHAR_50_200/', datasets='ICHAR')
    # datasets_users_record(dir=r'datasets/HASC_50_200/', datasets='HASC')


    
