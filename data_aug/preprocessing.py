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
from scipy.interpolate import interp1d
from collections import Counter
import random

sys.path.append(dirname(dirname(sys.path[0])))
sys.path.append(dirname(sys.path[0]))

from exceptions.exceptions import InvalidDatasetSelection
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
    'Shoaib': 10220,
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
    'HASC': 80,  # actually we have 64 users in total. But the largest user id is 79, set it to 80. +1 since it starts from 0. 
    'Myo': 40,
    'NinaPro': 10,
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
    'HASC': 6,
    'Myo': 7,
    'Myo_cda': 7,
    'NinaPro': 7,
    'NinaPro_cda': 7,
}

DevicesNum = {
    'HHAR': 6,
    'HASC': 24, # actually we have 18 devices in total. But the largest user id is 23, set it to 24. +1 since it starts from 0. 
}

PositionNum = {
    'Shoaib': 5,
}

HHAR_movement = ['stand', 'sit', 'walk', 'stairsup', 'stairsdown', 'bike']
ACT_Translated_labels = ['Downstairs', 'Upstairs', 'Walking', 'Running', 'Standing', 'Sitting']

# Common sequence ['still', 'walk', 'stairsup', 'stairsdown']
# Use -1 to mark unselected labels
HASC_LABEL_Translate = [-1, 3, 2, 1, -1, 0] # [‘jog’, ‘stairdown’, ‘stairup’, ‘move’, ‘jump’, ‘stay’]
HHAR_LABEL_Translate = [0, 0, 1, 2, 3, -1] # ['stand', 'sit', 'walk', 'stairsup', 'stairsdown', 'bike']
MotionSense_LABEL_Translate = [3, 2, 1, -1, 0, 0] # ['Downstairs', 'Upstairs', 'Walking', 'Running', 'Standing', 'Sitting']
Shoaib_LABEL_Translate = [1, 0, 0, -1, -1, 2, 3] # ["walking", "sitting", "standing", "jogging", "biking", "upstairs", "downstairs"] 


def label_alignment(label, dataset):
    if dataset == 'HASC':
        label[0] = HASC_LABEL_Translate[label[0]]
    elif dataset == 'HHAR':
        label[0] = HHAR_LABEL_Translate[label[0]]
    elif dataset == 'MotionSense':
        label[0] = MotionSense_LABEL_Translate[label[0]]
    elif dataset == 'Shoaib':
        label[0] = Shoaib_LABEL_Translate[label[0]]
    else:
        InvalidDatasetSelection()
    return label


samplingRate = 25
window = 100
train_ratio = 0.8
test_ratio = 0.2
tune_ratio = 0.2
test_num_of_user = 3



# MAX_INDEX = 9166
percent = [0.2, 0.5, 1, 2, 5, 10]
shot_num = [0, 1, 5, 10, 15, 20, 50, 100, 200, 500] # enlarge to 500
# shot_num=[0]


def preprocessing_plain_segmentation(dir, target_dir, val_portion=0.15, test_portion=0.25):
    num = fetch_instance_number_of_dataset(dir)
    idx = np.arange(num)
    test_num = int(num * test_portion)
    val_num = int(num * val_portion)
    np.random.shuffle(idx)
    
    test_instance = idx[:test_num]
    val_instance = idx[test_num: test_num + val_num]
    train_instance = idx[test_num + val_num:]
    
    print("Total Num: {}".format(num))
    print("Train num: {}".format(len(train_instance)))
    print("Val num: {}".format(len(val_instance)))
    print("Test num: {}".format(len(test_instance)))
    write_dataset(target_dir, train_instance, val_instance, test_instance)
    
    return
    
    

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


def preprocessing_dataset_cross_domain_val(dir, target_dir, dataset, test_portion=0.6, val_portion=0.15, tune_domain_portion=0.4, cross='users'):
    print(dataset)
    
    num = fetch_instance_number_of_dataset(dir)
    domain = []
    motion_label = []
    label_distribution = np.zeros(7)
    for i in range(num):
        sub_dir = dir + str(i) + '.npz'
        data = np.load(sub_dir, allow_pickle=True)
        motion = np.int32(data['add_infor'][0])
        if cross == 'users':
            domain.append(data['add_infor'][1])
        elif cross == 'devices':
            domain.append(data['add_infor'][2])
        elif cross == 'positions':
            domain.append(data['add_infor'][2])
        else:
            NotADirectoryError()
        motion_label.append(motion)
        label_distribution[motion] += 1
    
    print(label_distribution)
    # print(f"maximum of domains id {max(domain)}")
    domain_type = np.unique(domain)
    test_num = max(int(len(domain_type) * test_portion), 1)
    val_num = max(int(len(domain_type) * val_portion), 1)
    
    print(f"number of domains {domain_type}")
    np.random.shuffle(domain_type)
    domains_test_name = np.sort(domain_type[:test_num])
    # users_test_name = np.array(['e', 'i'])
    print(f"number of test domains {len(domains_test_name)}")

    domains_train_name = np.sort(domain_type[test_num+val_num:])
    # users_train_name = np.array(['a', 'd', 'f', 'g', 'h'])
    print(f"number of training domains {len(domains_train_name)}")
    
    domains_val_name = np.sort(domain_type[test_num:test_num+val_num])
    # users_val_name = np.array(['b', 'c'])
    print(f"number of validation domains {len(domains_val_name)}")

    train_num = [j for j in range(num) if domain[j] in domains_train_name]
    val_num =  [j for j in range(num) if domain[j] in domains_val_name]
    test_num = [j for j in range(num) if domain[j] in domains_test_name]

    write_dataset(target_dir, train_num, val_num, test_num)
    write_balance_tune_set(dir, target_dir, dataset, dataset_size=num, tune_domain_portion=tune_domain_portion, cross=cross)
    return


def turn_motion_label_to_idx(label, dataset):
    if dataset == 'HHAR':
        label = HHAR_movement.index(label)
    elif dataset == 'MotionSense':
        label = ACT_Translated_labels.index(label)
    else:
        pass
    return int(label)


def preprocessing_dataset_cross_dataset(dir, target_dir, dataset, test_portion=0.6, val_portion=0.15, tune_user_portion=0.4):
    print(dataset)
    
    num = fetch_instance_number_of_dataset(dir)
    u = []
    motion_label = []
    idx_record = []
    label_distribution = np.zeros(4)
    for i in range(num):
        sub_dir = dir + str(i) + '.npz'
        data = np.load(sub_dir, allow_pickle=True)
        motion = turn_motion_label_to_idx(data['add_infor'][0, LabelPosition[dataset]], dataset)
        motion = label_alignment([motion], dataset=dataset)[0]
        if motion == -1:  # if the motion is not considered, then we don't consider this window, then the user could not be considered.
            continue
        idx_record.append(i)
        u.append(data['add_infor'][0, UsersPosition[dataset]])
        motion_label.append(motion)
        label_distribution[motion] += 1
    
    print(label_distribution)
    print(max(u))
    user_type = np.unique(u)
    test_num = max(int(len(user_type) * test_portion), 1)
    val_num = max(int(len(user_type) * val_portion), 1)
    
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

    train_num = [idx_record[j] for j in range(len(motion_label)) if u[j] in users_train_name]
    val_num =  [idx_record[j] for j in range(len(motion_label)) if u[j] in users_val_name]
    test_num = [idx_record[j] for j in range(len(motion_label)) if u[j] in users_test_name]

    write_dataset(target_dir, train_num, val_num, test_num)
    write_balance_tune_set(dir, target_dir, dataset, dataset_size=len(motion_label), tune_domain_portion=tune_user_portion, cross_dataset=True)
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
    

def write_balance_tune_set(ori_dir, target_dir, dataset, dataset_size=None, if_percent=False, if_cross_user=True, tune_domain_portion=0.4, train_dir=None, cross='users', domain_for_tune=None):
    if train_dir is None:
        loc = target_dir + 'train_set' + '.npz'
    else:
        loc = train_dir + 'train_set' + '.npz'
    data = np.load(loc)
    train_set = data['train_set']
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    label = []
    domain = []

    for i in train_set:
        sub_dir = ori_dir + i
        data = np.load(sub_dir, allow_pickle=True)
        motion = data['add_infor'][0]
        if cross == 'dataset':
            motion = label_alignment([motion], dataset=dataset)[0]
        label.append(motion)

        if cross == 'users':
            domain.append(data['add_infor'][1])
        elif cross == 'devices':
            domain.append(data['add_infor'][2])
        elif cross == 'positions':
            domain.append(data['add_infor'][2])
        else:
            NotADirectoryError()

    label = np.array(label)
    label_type = np.unique(label)
    label_type_num = len(label_type)
    print(f"type of labels {label_type}")
    if cross == 'dataset':
        assert label_type_num == 4
    else:
        assert label_type_num == ClassesNum[dataset]

    while True:
        domain_type = np.unique(domain)
        domain_selected_num = np.max([int(len(domain_type) * tune_domain_portion), 1])


        np.random.shuffle(domain_type)

        if domain_for_tune is None:
            domain_selected = domain_type[:domain_selected_num]
        else:
            domain_selected = domain_for_tune

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
                
                for i in label_type:
                    idx = np.argwhere((label == i)).squeeze()
                    if cross:
                        idx_domain_selected = []
                        for j in idx:
                            if domain[j] in domain_selected:
                                idx_domain_selected.append(j)
                        idx = np.array(idx_domain_selected)
                        
                    np.random.shuffle(idx)
                    if len(idx) == 0:
                        irreasonable_segmentation = 1
                        break
                    if label_per_class == 0:
                        tune_set.extend(train_set[idx])
                        counter.append(len(list(idx)))
                    else:
                        tune_set.extend(train_set[idx[:label_per_class]])
                        counter.append(len(list(idx[:label_per_class])))
                
                if irreasonable_segmentation:
                    break

                print(f"Shot num {label_per_class}: {len(tune_set)}")
                loca = target_dir + 'tune_set_' + str(label_per_class).replace('.', '_') + '.npz'
                np.savez(loca, tune_set=tune_set)
            
            if irreasonable_segmentation == 0:
                break

    print(f"type of domains {domain_type}")
    print(f"motion classes {label_type_num}, total train num {len(label)}, total domain num {len(domain_type)}, domain {domain_selected} provides label")
    return


def datasets_shot_record(datasets, set_type='tune_set', version='shot0', shot=10):
    data_dir = r'./datasets/' + datasets + '/'
    dir = r'./datasets/' + datasets + '_' + version + '/'
    tune_dir = dir + set_type + '_' + str(shot) + '.npz'
    data = np.load(tune_dir, allow_pickle=True)
    data = data[set_type]
    label_type = []
    label = []
    print(len(data))
    for i in data:
        sub_dir = data_dir + str(i)
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
        preprocessing_dataset_cross_domain_val(dir=path_save, dataset='HHAR', dataset_size=13047, test_num=t, cross='users')
    return


def extract_and_seg_hhar(path_save, dataset, window_time, seq_len, version, test_num):
    num = preprocess_hhar(DATASET_PATH, path_save, version=version, window_time=window_time, seq_len=seq_len)  # use jump to control overlap.
    preprocessing_dataset_cross_domain_val(dir=path_save, dataset=dataset, dataset_size=num, test_num=test_num, cross='users')
    return

DATASET_PATH = r'./original_dataset/hhar/'

def new_segmentation_for_user(seg_types=5, seed=940):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    dataset_name = ["NinaPro_cda"]
    # dataset_name = ["Shoaib"]
    for i in range(seg_types):
        for dataset in dataset_name:
            preprocessing_dataset_cross_domain_val(dir=f'datasets/{dataset}/', target_dir=f"datasets/{dataset}_shot{i}/", dataset=dataset, cross='users')

    return

def new_segmentation_for_dataset(seg_types=5, seed=940):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    dataset_name = ["HASC", "HHAR", "Shoaib", "MotionSense"]
    # dataset_name = ["Shoaib"]
    for i in range(seg_types):
        for dataset in dataset_name:
            preprocessing_dataset_cross_dataset(dir=f'datasets/{dataset}/', target_dir=f"datasets/{dataset}_cross_dataset{i}/", dataset=dataset)

    return

def new_segmentation_for_positions(seg_types=5, seed=940):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    dataset = "Shoaib"
    for i in range(seg_types):
        preprocessing_dataset_cross_domain_val(dir=f'datasets/{dataset}/', target_dir=f"datasets/{dataset}_cp{i}/", test_portion=0.4, val_portion=0.2, tune_domain_portion=0.5, dataset=dataset, cross='positions')
    return


def new_segmentation_for_devices(seg_types=1, seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    dataset = "HASC"
    for i in range(seg_types):
        preprocessing_dataset_cross_domain_val(dir=f'datasets/{dataset}/', target_dir=f"datasets/{dataset}_cd_test{i}/", test_portion=0.6, val_portion=0.15, tune_domain_portion=0.4, dataset=dataset, cross='devices')
    return

    
def new_segmentation_for_domain_shift_visual(seed=940, seg_types=5):
    random.seed(seed)
    np.random.seed(seed)
    dataset = 'HHAR'
    
    for i in range(seg_types):
        # preprocessing_plain_segmentation(f'datasets/{dataset}/', f"datasets/{dataset}_train65_alltune_plain_v{i}/", val_portion=0.15, test_portion=0.20)
        preprocessing_dataset_cross_domain_val(dir=f'datasets/{dataset}/', target_dir=f"datasets/{dataset}_train65_alltune_cross_v{i}/", dataset=dataset, val_portion=0.15, test_portion=0.20, tune_user_portion=1, cross='users')
        
        # preprocessing_plain_segmentation(f'datasets/{dataset}/', f"datasets/{dataset}_train60_supervised_plain/", val_portion=0.15, test_portion=0.25)
        # preprocessing_dataset_cross_domain_val(dir=f'datasets/{dataset}/', target_dir=f"datasets/{dataset}_train60_supervised_cross/", dataset=dataset, val_portion=0.15, test_portion=0.25)
        
        # preprocessing_plain_segmentation(f'datasets/{dataset}/', f"datasets/{dataset}_train50_supervised_plain/", val_portion=0.15, test_portion=0.35)
        # preprocessing_dataset_cross_domain_val(dir=f'datasets/{dataset}/', target_dir=f"datasets/{dataset}_train50_supervised_cross/", dataset=dataset, val_portion=0.15, test_portion=0.35)
        
        # preprocessing_plain_segmentation(f'datasets/{dataset}/', f"datasets/{dataset}_train45_alltune_plain_v{i}/", val_portion=0.15, test_portion=0.40)
        preprocessing_dataset_cross_domain_val(dir=f'datasets/{dataset}/', target_dir=f"datasets/{dataset}_train45_alltune_cross_v{i}/", dataset=dataset, val_portion=0.15, test_portion=0.40, tune_user_portion=1, cross='users')

        # preprocessing_plain_segmentation(f'datasets/{dataset}/', f"datasets/{dataset}_train25_alltune_plain_v{i}/", val_portion=0.15, test_portion=0.60)
        preprocessing_dataset_cross_domain_val(dir=f'datasets/{dataset}/', target_dir=f"datasets/{dataset}_train25_alltune_cross_v{i}/", dataset=dataset, val_portion=0.15, test_portion=0.60, tune_user_portion=1, cross='users')


def new_tune_segmentation_with_different_portion(seed=940, seg_type=1):
    random.seed(seed)
    np.random.seed(seed)
    # dataset_name = ["HASC", "HHAR", "Shoaib", "MotionSense"]
    dataset_name = ["HASC"]
    # dataset_name = ["HHAR"]
    tune_portion = [0.6, 0.8, 1.0]
    for i in range(seg_type):
        for portion in tune_portion:
            for dataset in dataset_name:
                write_balance_tune_set(ori_dir=f'datasets/{dataset}/', train_dir=f'datasets/{dataset}_cd{i}/', target_dir=f'datasets/{dataset}_cd_tune_portion_{int(portion*100)}_shot{i}/', dataset=dataset, tune_domain_portion=portion, cross='devices')
    return

def random_split(dir, cross_domain_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    train_loc = cross_domain_dir + 'train_set.npz'
    val_loc = cross_domain_dir + 'val_set.npz'
    test_loc = cross_domain_dir + 'test_set.npz'
    cross_train_set, cross_val_set, cross_test_set = np.load(train_loc), np.load(val_loc), np.load(test_loc)
    cross_train_set, cross_val_set, cross_test_set = cross_train_set['train_set'], cross_val_set['val_set'], cross_test_set['test_set']
    train_len = len(cross_train_set)
    val_len = len(cross_val_set)
    num = fetch_instance_number_of_dataset(dir)
    samples = np.arange(num)
    np.random.shuffle(samples)
    train_num = samples[:train_len]
    val_num = samples[train_len:train_len+val_len]
    test_num = samples[train_len+val_len:]
    print(f"the total number is {num}, divided into train {len(train_num)}, val {len(val_num)}, test {len(test_num)}")
    write_dataset(target_dir, train_num, val_num, test_num)
    return


def cmp_split():
    # dataset_1 = 'NinaPro_cda'
    # dataset_2 = 'NinaPro'

    dataset_1 = 'Myo_cda'
    dataset_2 = 'Myo'

    for v in range(5):
        dir = 'datasets/' + dataset_1 + '_shot' + str(v)
        domain_split_1 = get_split_infor(dir)

        dir = 'datasets/' + dataset_2 + '_shot' + str(v)
        domain_split_2 = get_split_infor(dir)

        print(domain_split_1)
        print(domain_split_2)


def generate_split_for_cda_based_on_previous_split():
    # dataset_1 = 'NinaPro_cda'
    # dataset_2 = 'NinaPro'
    
    dataset_1 = 'Myo_cda'
    dataset_2 = 'Myo'

    for v in range(5):
        dir = 'datasets/' + dataset_2 + '_shot' + str(v)
        split_v = get_split_infor(dir)
        preprocessing_dataset_cross_domain_based_on_existing_split(split_v, dir=f'datasets/{dataset_1}/', 
                                                                   target_dir=f"datasets/{dataset_1}_shot{v}/", 
                                                                   dataset=dataset_1, cross='users')


def get_split_infor(dir):
    train = 'train_set.npz'
    val = 'val_set.npz'
    tune = 'tune_set_500.npz'
    test = 'test_set.npz'
    
    split_list = [train, val, tune, test]
    
    domain_split = []
    for i in split_list:
        domain = []
        i_dir = dir + '/' + i
        id_list = np.load(i_dir, allow_pickle=True)
        if 'tune' in i:
            id_list = id_list[i[:-8]]
        else:
            id_list = id_list[i[:-4]]
        for j in id_list:
            d = np.load(dir[:-6]+'/'+j, allow_pickle=True)
            domain.append(d['add_infor'][1])
        
        domain = np.unique(np.hstack(domain))
        
        domain_split.append(domain)
        # print(domain)
    
    return domain_split

def preprocessing_dataset_cross_domain_based_on_existing_split(split, dir, target_dir, dataset, tune_domain_portion=0.4, cross='users'):
    domains_train_name = split[0]
    domains_val_name = split[1]
    domains_test_name = split[3]

    num = fetch_instance_number_of_dataset(dir)
    domain = []
    for i in range(num):
        sub_dir = dir + str(i) + '.npz'
        data = np.load(sub_dir, allow_pickle=True)
        motion = np.int32(data['add_infor'][0])
        if cross == 'users':
            domain.append(data['add_infor'][1])
        elif cross == 'devices':
            domain.append(data['add_infor'][2])
        elif cross == 'positions':
            domain.append(data['add_infor'][2])
        else:
            NotADirectoryError()
    
    train_num = [j for j in range(num) if domain[j] in domains_train_name]
    val_num =  [j for j in range(num) if domain[j] in domains_val_name]
    test_num = [j for j in range(num) if domain[j] in domains_test_name]

    write_dataset(target_dir, train_num, val_num, test_num)
    write_balance_tune_set(dir, target_dir, dataset, dataset_size=num, tune_domain_portion=tune_domain_portion, cross=cross, domain_for_tune=split[2])


if __name__ == '__main__':
    # datasets_shot_record(datasets='HASC', version='s1', shot=100)
    # new_segmentation_for_positions(seg_types=5)
    # new_segmentation_for_devices(seg_types=1)
    # new_segmentation_for_user(seg_types=5)
    generate_split_for_cda_based_on_previous_split()
    cmp_split()
    # new_tune_segmentation_with_different_portion(seed=940, seg_type=5)
    # dataset='Myo'
    # preprocessing_dataset_cross_domain_val(dir=f'datasets/{dataset}/', target_dir=f"datasets/{dataset}_shot0/", test_portion=0.6, val_portion=0.15, tune_domain_portion=0.4, dataset=dataset, cross='users')
    
    # write_balance_tune_set(ori_dir=f'datasets/{dataset}/', train_dir="", target_dir=f'datasets/{dataset}_shot0/', dataset=dataset, cross='users')
    # preprocessing_dataset_cross_domain_val(dir=f'datasets/{dataset}/', target_dir=f"datasets/{dataset}_domain_shift/", dataset=dataset, cross='users')
    # random_split(dir=f'datasets/{dataset}/', cross_domain_dir=f'datasets/HHAR_train25_supervised_cross/', target_dir=f'datasets/HHAR_train25_supervised_random/')