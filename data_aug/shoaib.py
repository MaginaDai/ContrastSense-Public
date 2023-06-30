# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 28/4/2021
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : shoaib.py
# @Description : https://www.mdpi.com/1424-8220/14/6/10146

import os
import sys
import pdb
import numpy as np
import pandas as pd
from torch.utils.data import random_split
from os.path import dirname
sys.path.append(dirname(sys.path[0]))
from data_aug.preprocessing import preprocessing_dataset_cross_person
from data_aug.MotionSense_Prep import percent

DATASET_PATH = r'./original_dataset/Shoaib'
ACT_LABELS = ["walking", "sitting", "standing", "jogging", "biking", "upstairs", "downstairs"]
SAMPLE_WINDOW = 20


def label_name_to_index(label_names):
    label_index = np.zeros(label_names.size)
    for i in range(len(ACT_LABELS)):
        ind = label_names == ACT_LABELS[i]
        # print(np.sum(ind))
        label_index[ind] = i
    return label_index


def down_sample(data, window_target):
    window_sample = window_target * 1.0 / SAMPLE_WINDOW
    result = []
    if window_sample.is_integer():
        window = int(window_sample)
        for i in range(0, len(data), window):
            slice = data[i: i + window, :]
            result.append(np.mean(slice, 0))
    else:
        window = int(window_sample)
        remainder = 0.0
        i = 0
        while 0 <= i + window + 1 <= data.shape[0]:
            remainder += window_sample - window
            if remainder >= 1:
                remainder -= 1
                slice = data[i: i + window + 1, :]
                # print('i: %d, window: %d, start: %d, end: %d' % (i, window, start, end))
                result.append(np.mean(slice, 0))
                i += window + 1
            else:
                slice = data[i: i + window, :]
                result.append(np.mean(slice, 0))
                # print('i: %d, window: %d, start: %d, end: %d' % (i, window +  1, start, end))
                i += window
    return np.array(result)


def preprocess(path, path_save, target_window=20, seq_len=20, position_num=5):
    num = 0
    time_idx = 0
    for root, dirs, files in os.walk(path):
        for f in range(len(files)):
            if 'Participant' in files[f]:
                exp = pd.read_csv(os.path.join(root, files[f]), skiprows=1)
                labels_activity = exp.iloc[:, -1].to_numpy()
                labels_activity = label_name_to_index(labels_activity)
                for a in range(len(ACT_LABELS)):
                    exp_act = exp.iloc[labels_activity == a, :]
                    for i in range(position_num):
                        index = np.array([1, 2, 3, 7, 8, 9, 10, 11, 12]) + i * 14
                        exp_pos = exp_act.iloc[:, index].to_numpy(dtype=np.float32)
                        print("User-%s, activity-%s, position-%d: num-%d" % (files[f][-5], ACT_LABELS[a], i, exp_pos.shape[0]))
                        if exp_pos.shape[0] > 0:
                            exp_pos_down = down_sample(exp_pos, target_window)
                            if exp_pos_down.shape[0] < seq_len:
                                continue
                            else:
                                # [int(add_infor[0, -2]), int(add_infor[0, UsersPosition[self.datasets_name]]), int(add_infor[0, -3])]
                                sensor_down = exp_pos_down[:exp_pos_down.shape[0] // seq_len * seq_len, :]
                                sensor_down = sensor_down.reshape(sensor_down.shape[0] // seq_len, seq_len, sensor_down.shape[1])
                                sensor_label = np.array([a, int(files[f][-5]), i]) # [motion, user, position]
                                if sensor_label[1] == 0:
                                    print("now")
                                for m in range(sensor_down.shape[0]):
                                    acc_new = sensor_down[m][:, 0:3]
                                    gyro_new = sensor_down[m][:, 3:6]
                                    loc = path_save + str(num) + '.npz'
                                    np.savez(loc, acc=acc_new, gyro=gyro_new, add_infor=np.array([a, int(files[f][-5]), i, time_idx]))
                                    time_idx += 1
                                    num += 1
                                    print(time_idx)
                                time_idx += 1000
    # split_dataset(num=num - 1)  # num-1 for the last void one.
    # preprocessing_dataset_cross_person(dir=path_save, dataset='Shoaib')
    print("done")
    return


def split_dataset(num):
    train_set_len = int(num * 0.8)
    val_set_len = int(num * 0.1)
    train_num, val_num, test_num = random_split(range(0, num),
                                                [train_set_len, val_set_len,
                                                 num - train_set_len - val_set_len])
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
    np.savez(os.path.join(path_save, 'train_set' + '.npz'), train_set=train_set)
    np.savez(os.path.join(path_save, 'val_set' + '.npz'), val_set=val_set)
    np.savez(os.path.join(path_save, 'test_set' + '.npz'), test_set=test_set)

    loc = path_save + 'train_set' + '.npz'
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
        loc = path_save + 'tune_set_' + str(per).replace('.', '_') + '.npz'
        np.savez(loc, tune_set=tune_set)


if __name__ == '__main__':
    path_save = r'./datasets/Shoaib_time/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    preprocess(DATASET_PATH, path_save, seq_len=200)
