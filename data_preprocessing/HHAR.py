import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from collections import Counter
from torch.utils.data import random_split
import os
from os.path import dirname
import sys

sys.path.append(dirname(dirname(sys.path[0])))
sys.path.append(dirname(sys.path[0]))

movement = ['stand', 'sit', 'walk', 'stairsup', 'stairsdown', 'bike']
devices = ['nexus4_1', 'nexus4_2', 's3_1', 's3_2', 's3mini_1', 's3mini_2', 'samsungold_1', 'samsungold_2']
models = ['nexus4', 's3', 's3mini', 'samsungold']
users = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
uot_movement = ['Standing', 'Sitting', 'Walking', 'Upstairs', 'Downstairs',  'Running']

samplingRate = 25
window = 100
train_ratio = 0.8
test_ratio = 0.2
tune_ratio = 0.2
test_num_of_user = 3


# MAX_INDEX = 9166
percent = [0.2, 0.5, 1, 2, 5, 10]


def extract_sensor(data, time_index, time_tag, window_time):
    index = time_index
    while index < len(data) and abs(data.iloc[index]['Creation_Time'] - time_tag) < window_time:
        index += 1
    if index == time_index:
        return None, index
    else:
        data_slice = data.iloc[time_index:index]
        if data_slice['User'].unique().size > 1 or data_slice['gt'].unique().size > 1:
            return None, index
        else:
            data_sensor = data_slice[['x', 'y', 'z']].to_numpy()
            sensor = np.mean(data_sensor, axis=0)
            label = data_slice[['User', 'Device', 'gt']].iloc[0].values
            return np.concatenate([sensor, label]), index


def transform_to_index(label, print_label=False):
    labels_unique = np.unique(label)
    if print_label:
        print(labels_unique)
    for i in range(labels_unique.size):
        label[label == labels_unique[i]] = i


def separate_data_label(data_raw):
    labels = data_raw[:, :, -3:].astype(np.str)
    transform_to_index(labels[:, :, 0])
    transform_to_index(labels[:, :, 1], print_label=True)
    transform_to_index(labels[:, :, 2], print_label=True)
    data = data_raw[:, :, :6].astype(np.float)
    labels = labels.astype(np.float)
    return data, labels


# 'Index', 'Arrival_Time', 'Creation_Time', 'x', 'y', 'z', 'User', 'Model', 'Device', 'gt'
def preprocess_hhar(path, path_save, version, window_time=50, seq_len=40, jump=0):
    accs = pd.read_csv(path + '/Phones_accelerometer.csv')
    gyros = pd.read_csv(path + '/Phones_gyroscope.csv')  # nrows=200000
    time_tag = min(accs.iloc[0, 2], gyros.iloc[0, 2])
    time_index = [0, 0] # acc, gyro
    window_num = 0
    data_temp = []
    num = 0
    old_user = 0
    current_user = 0
    time_idx = 0
    while time_index[0] < len(accs) and time_index[1] < len(gyros):
        acc, time_index_new_acc = extract_sensor(accs, time_index[0], time_tag, window_time=window_time * pow(10, 6))
        gyro, time_index_new_gyro = extract_sensor(gyros, time_index[1], time_tag, window_time=window_time * pow(10, 6))
        time_index = [time_index_new_acc, time_index_new_gyro]
        if acc is not None and gyro is not None and np.all(acc[-3:] == gyro[-3:]):
            time_tag += window_time * pow(10, 6)
            window_num += 1
            data_temp.append(np.concatenate([acc[:-3], gyro[:-3], acc[-3:]]))
            if window_num == seq_len:
                data_raw = np.array(data_temp)
                add_infor=data_raw[0, 6:] # [users, devices, motion]
                if num == 0:
                    old_user = users.index(add_infor[-3])
                current_user = users.index(add_infor[-3])
                if current_user == old_user:
                    time_idx += 1
                else:
                    print(old_user)
                    print(current_user)
                    time_idx += 1e3
                    old_user = current_user
                add_infor=np.array([movement.index(add_infor[-1]), users.index(add_infor[-3]), devices.index(add_infor[-2]), time_idx]) # [motion, users, devices]
                if num % 100 == 0:
                    print(num)
                
                np.savez(os.path.join(path_save, str(num) + '.npz'), acc=data_raw[:, 0:3], gyro=data_raw[:, 3:6], add_infor=add_infor)
                num += 1
                if jump == 0:
                    data_temp.clear()
                    window_num = 0
                else:
                    data_temp = data_temp[-jump:]
                    window_num -= jump
        else:
            if window_num > 0:
                data_temp.clear()
                window_num = 0
            if time_index[0] < len(accs) and time_index[1] < len(gyros):
                time_tag = min(accs.iloc[time_index[0], 2], gyros.iloc[time_index[1], 2])
            else:
                break

    # num = 9167
    num -= 1
    return num


def label_translate(source_dir, target_dir):
    file_name_list = [file for file in os.listdir(source_dir) if 'set' not in file]
    num = len(file_name_list)
    print(num)
    for idx in np.arange(num):
        loc = os.path.join(source_dir, f'{idx}.npz')
        sample = np.load(loc, allow_pickle=True)
        acc, gyro, add_infor = sample['acc'], sample['gyro'], sample['add_infor']
        add_infor = np.array([movement.index(add_infor[0, -1]), users.index(add_infor[0, -3]), models.index(add_infor[0, -2])]) # [motion, users, devices]
        np.savez(os.path.join(target_dir, str(idx) + '.npz'), acc=acc, gyro=gyro, add_infor=add_infor)
    return

DATASET_PATH = r'./original_dataset/hhar/'
if __name__ == '__main__':
    path_save = r'./datasets/HHAR/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    num = preprocess_hhar(DATASET_PATH, path_save, version='test', window_time=20, seq_len=200)  # use jump to control overlap.

