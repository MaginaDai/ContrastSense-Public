from cProfile import label
from gettext import find
from itertools import count
import os
from os.path import dirname
import pdb
import sys
from numpy.core.memmap import dtype
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from scipy.interpolate import interp1d
from collections import Counter
import random

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

UsersNum = {
    'HHAR': 9,
    'MotionSense': 24,
    'Shoaib': 10,
    'UCI': 30,
    'ICHAR': 10,
    'HASC': 306,  # actually we have 288 users in total. But the largest user id is 305, set it to 306. +1 since it starts from 0. 
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
shot_num = [1, 5, 10, 15, 20, 50]


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


def fetch_instance_number_of_dataset(dir):
    file_name_list = [file for file in os.listdir(dir) if 'set' not in file]
    return len(file_name_list)


def preprocessing_dataset_cross_person_val(dir, target_dir, dataset, test_portion=0.6, val_portion=0.15):
    print(dataset)
    
    num = fetch_instance_number_of_dataset(dir)
    u = []
    # label_distribution = np.zeros(6)
    for i in range(num):
        sub_dir = dir + str(i) + '.npz'
        data = np.load(sub_dir, allow_pickle=True)
        u.append(data['add_infor'][0, UsersPosition[dataset]])
        # label_distribution[int(data['add_infor'][0, -1])] += 1
    
    # print(label_distribution)
    print(max(u))
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

    write_dataset(target_dir, train_num, val_num, test_num)
    write_balance_tune_set(dir, target_dir, dataset, dataset_size=num)
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
    if not os.path.exists(dir):
        os.makedirs(dir)
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
        # np.savez(loc, tune_set=tune_set)
    return


def write_balance_tune_set(ori_dir, target_dir, dataset, dataset_size=None, if_percent=False, if_cross_user=True):
    loc = target_dir + 'train_set' + '.npz'
    data = np.load(loc)
    train_set = data['train_set']
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    label = []
    user = []

    for i in train_set:
        sub_dir = ori_dir + i
        data = np.load(sub_dir, allow_pickle=True)
        label.append(data['add_infor'][0, LabelPosition[dataset]])
        if dataset == 'Shoaib':
            user.append(int(data['add_infor'][0, UsersPosition[dataset]]))
        else:
            user.append(data['add_infor'][0, UsersPosition[dataset]])

    label = np.array(label)
    label_type = np.unique(label)
    label_type_num = len(label_type)
    print(label_type) 
    assert label_type_num == ClassesNum[dataset]

    while True:
        user_type = np.unique(user)
        user_selected_num = np.max([int(len(user_type) * 0.4), 1])


        np.random.shuffle(user_type)
        user_selected = user_type[:user_selected_num]

        if dataset_size is None:
            set_size = fetch_instance_number_of_dataset(ori_dir)
        else:
            set_size = dataset_size
        if if_percent:
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
                loca = target_dir + 'tune_set_' + str(per).replace('.', '_') + '.npz'
                np.savez(loca, tune_set=tune_set)
        else:
            irreasonable_segmentation = 0
            for label_per_class in shot_num:
                tune_set = []
                counter = []
                if label_per_class < 1:
                    label_per_class = 1  # at least one label for each class
                    print("at least one sample per class")
                
                for i in label_type:
                    idx = np.argwhere((label == i)).squeeze()
                    if if_cross_user:
                        idx_user_selected = []
                        for j in idx:
                            if user[j] in user_selected:
                                idx_user_selected.append(j)
                        idx = np.array(idx_user_selected)
                    np.random.shuffle(idx)
                    if len(idx) == 0:
                        irreasonable_segmentation = 1
                        break
                    tune_set.extend(train_set[idx[:label_per_class]])
                    counter.append(len(list(idx[:label_per_class])))
                
                if irreasonable_segmentation:
                    break

                print(f"Shot num {label_per_class}: {len(tune_set)}")
                loca = target_dir + 'tune_set_' + str(label_per_class).replace('.', '_') + '.npz'
                np.savez(loca, tune_set=tune_set)
            
            if irreasonable_segmentation == 0:
                break

    print(user_type)
    print(f"motion classes {label_type_num}, total train num {len(label)}, total user num {len(user_type)}, user {user_selected} provides label")
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

def new_segmentation_for_user(seg_types=5, seed=940):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    dataset_name = ["HASC", "HHAR", "Shoaib", "MotionSense"]
    for i in range(seg_types):
        for dataset in dataset_name:
            preprocessing_dataset_cross_person_val(dir=f'datasets/{dataset}/', target_dir=f"datasets/{dataset}_shot{i}/", dataset=dataset)

    return

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

    # preprocessing_dataset_cross_person_val(dir=r'datasets/HHAR/', target_dir=r'datasets/HHAR_shot/', dataset='HHAR')
    # preprocessing_dataset_cross_person_val(dir=r'datasets/MotionSense/', target_dir=r'datasets/MotionSense_shot/', dataset='MotionSense')
    # preprocessing_dataset_cross_person_val(dir=r'datasets/Shoaib/', target_dir=r'datasets/Shoaib_shot/', dataset='Shoaib')
    # preprocessing_dataset_cross_person_val(dir=r'datasets/HASC/', target_dir=r'datasets/HASC_shot/', dataset='HASC')
    # preprocessing_dataset_cross_person_val(dir=r'datasets/UCI/', target_dir=r'datasets/HHAR_shot/', dataset='UCI')
    # preprocessing_dataset_cross_person_val(dir=r'datasets/ICHAR/', target_dir=r'datasets/HHAR_shot/', dataset='ICHAR')
 
    # datasets_shot_record(dir=r'datasets/HHAR_50_200/', datasets='HHAR')

    # datasets_users_record(dir=r'datasets/HHAR_50_200/', datasets='HHAR')
    # datasets_users_record(dir=r'datasets/MotionSense_50_200/', datasets='MotionSense')
    # datasets_users_record(dir=r'datasets/Shoaib_50_200/', datasets='Shoaib')
    # datasets_users_record(dir=r'datasets/UCI_50_200/', datasets='UCI')
    # datasets_users_record(dir=r'datasets/ICHAR_50_200/', datasets='ICHAR')
    # datasets_users_record(dir=r'datasets/HASC_50_200/', datasets='HASC')


    # write_balance_tune_set(ori_dir=r'datasets/HHAR/', target_dir=r'datasets/HHAR_shot/', dataset='HHAR')
    # write_balance_tune_set(ori_dir=r'datasets/MotionSense_50_200/', target_dir=r'datasets/MotionSense_50_200_shot/', dataset='MotionSense')
    # write_balance_tune_set(ori_dir=r'datasets/Shoaib_50_200/', target_dir=r'datasets/Shoaib_50_200_shot/', dataset='Shoaib')
    # write_balance_tune_set(ori_dir=r'datasets/HASC_50_200/', target_dir=r'datasets/HASC_50_200_shot/', dataset='HASC')

    # new_segmentation_for_user(seg_types=5)
    
    dir = r'datasets/HHAR/'
    preprocessing_dataset_cross_person_val(dir, target_dir=r'datasets/HHAR_shot_portion35/', dataset='HHAR', test_portion=0.5, val_portion=0.15)
    preprocessing_dataset_cross_person_val(dir, target_dir=r'datasets/HHAR_shot_portion50/', dataset='HHAR', test_portion=0.35, val_portion=0.15)
    preprocessing_dataset_cross_person_val(dir, target_dir=r'datasets/HHAR_shot_portion60/', dataset='HHAR', test_portion=0.25, val_portion=0.15)
    preprocessing_dataset_cross_person_val(dir, target_dir=r'datasets/HHAR_shot_portion75/', dataset='HHAR', test_portion=0.10, val_portion=0.15)
