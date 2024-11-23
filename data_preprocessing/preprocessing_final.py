import os, sys, random
from os.path import dirname
import torch
import pandas as pd
import numpy as np

sys.path.append(dirname(dirname(sys.path[0])))
sys.path.append(dirname(sys.path[0]))

from exceptions.exceptions import InvalidDatasetSelection


HASC_LABEL_Translate = [-1, 3, 2, 1, -1, 0] # [‘jog’, ‘stairdown’, ‘stairup’, ‘move’, ‘jump’, ‘stay’]
HHAR_LABEL_Translate = [0, 0, 1, 2, 3, -1] # ['stand', 'sit', 'walk', 'stairsup', 'stairsdown', 'bike']
MotionSense_LABEL_Translate = [3, 2, 1, -1, 0, 0] # ['Downstairs', 'Upstairs', 'Walking', 'Running', 'Standing', 'Sitting']
Shoaib_LABEL_Translate = [1, 0, 0, -1, -1, 2, 3] # ["walking", "sitting", "standing", "jogging", "biking", "upstairs", "downstairs"] 

ClassesNum = {
    'HHAR': 6,
    'MotionSense': 6,
    'Shoaib': 7,
    'HASC': 6
}

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


def fetch_instance_number_of_dataset(dir):
    file_name_list = [file for file in os.listdir(dir) if 'set' not in file]
    return len(file_name_list)


def preprocessing_dataset_cross_person_val(dir, target_dir, dataset, test_portion=0.75, val_portion=0.20, tune_user_portion=0.4):
    print(dataset)
    
    num = fetch_instance_number_of_dataset(dir)
    u = []
    motion_label = []
    label_distribution = np.zeros(7)
    for i in range(num):
        sub_dir = dir + str(i) + '.npz'
        data = np.load(sub_dir, allow_pickle=True)
        motion = data['add_infor'][0]
        u.append(data['add_infor'][1])
        motion_label.append(motion)
        label_distribution[motion] += 1
    
    print(max(u))
    user_type = np.unique(u)
    test_num = max(int(len(user_type) * test_portion), 1)
    
    print(f"user type {len(user_type)}")
    np.random.shuffle(user_type)
    users_test_name = np.sort(user_type[:test_num])
    users_train_name = np.sort(user_type[test_num:])

    print(f"source num: {len(users_train_name)}")
    print(f"target num: {len(users_test_name)}")

    train_instance = [j for j in range(num) if u[j] in users_train_name]
    
    val_num = int(val_portion * len(train_instance))
    np.random.shuffle(train_instance)
    val_instance = train_instance[:val_num]
    potential_tune_instance = train_instance[val_num:]

    test_instance = [j for j in range(num) if u[j] in users_test_name]

    write_dataset(target_dir, train_instance, val_instance, test_instance)
    write_tune_set(dir, target_dir, dataset, dataset_size=num, tune_user_portion=tune_user_portion)
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

    print(f"train set len: {len(train_set)}")
    print(f"val set len: {len(val_set)}")
    print(f"test set len: {len(test_set)}")
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.savez(os.path.join(dir, 'train_set' + '.npz'), train_set=train_set)
    np.savez(os.path.join(dir, 'val_set' + '.npz'), val_set=val_set)
    np.savez(os.path.join(dir, 'test_set' + '.npz'), test_set=test_set)    
    return

sample_num = [60, 100, 200, 500, 1000, 'full']

def write_tune_set(ori_dir, target_dir, dataset, dataset_size=None, if_cross_user=True, tune_user_portion=0.4, cross_dataset=False):
    data = np.load(target_dir + 'train_set' + '.npz')
    train_set = data['train_set']
    
    val_data = np.load(target_dir + 'val_set' + '.npz')
    val_set = val_data['val_set']
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    label = []
    user = []
    
    # del validation data in the training set to guarantee the data in the tuning set are not included in the validation set.
    train_except_val = [i for i in train_set if i not in val_set]
    train_set = np.array(train_except_val)
    
    for i in train_set:
        sub_dir = ori_dir + i
        data = np.load(sub_dir, allow_pickle=True)
        motion = data['add_infor'][0]
        if cross_dataset:
            motion = label_alignment([motion], dataset=dataset)[0]
        label.append(motion)
        user.append(data['add_infor'][1])

    label = np.array(label)
    label_type = np.unique(label)
    label_type_num = len(label_type)
    print(label_type)

    if cross_dataset:
        assert label_type_num == 4
    else:
        assert label_type_num == ClassesNum[dataset]

    user_type = np.unique(user)
    user_selected_num = np.max([int(len(user_type) * tune_user_portion), 1])

    while True:
        np.random.shuffle(user_type)
        user_selected = user_type[:user_selected_num]
        
        selected_set = np.array([n for i, n in enumerate(train_set) if user[i] in user_selected])
        label_set = np.array([label[i] for i, n in enumerate(train_set) if user[i] in user_selected])
        if len(np.unique(label_set)) == label_type_num:
            break

    for num in sample_num:
        if num == 'full':
            tune_set = selected_set
            label_of_tune = label_set
        else:
            tune_set = []
            for i in label_type:  # sample 1 instance from each labels to guarantee each class has at least one label
                idx = np.argwhere((label_set == i)).squeeze()
                same_label_set = selected_set[idx]
                np.random.shuffle(same_label_set)
                tune_set.append(same_label_set[0])

            selected_set_prune = [i for i in selected_set if i not in tune_set]
            np.random.shuffle(selected_set_prune)
            tune_set.extend(selected_set_prune[:num-label_type_num])

            tune_set.sort()
            label_of_tune = []
            for i in tune_set:
                idx = np.argwhere(selected_set == i).squeeze()
                label_of_tune.append(label_set[idx])
        
        count=[]
        for i in label_type:
            count.append(len(np.argwhere(label_of_tune == i)))  # add 1 for the pre-selected one from each class

        print(f"Tune num {num}: {len(tune_set)}, each class {count} sum {sum(count)}")
        loca = target_dir + 'tune_set_' + str(num) + '.npz'
        np.savez(loca, tune_set=tune_set)

    print(user_type)
    print(f"motion classes {label_type_num}, total train num {len(label)}, total user num {len(user_type)}, user {user_selected} provides label")
    return


def new_segmentation_for_user(seg_types=5):
    dataset_name = ["HASC", "HHAR", "Shoaib", "MotionSense"]
    # dataset_name = ["Shoaib"]
    for i in range(seg_types):
        for dataset in dataset_name:
            preprocessing_dataset_cross_person_val(dir=f'datasets/{dataset}_50_200/', target_dir=f"datasets/{dataset}_s{i}/", dataset=dataset)

    return


if __name__ == '__main__':
    seed=940
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    new_segmentation_for_user(seg_types=5)