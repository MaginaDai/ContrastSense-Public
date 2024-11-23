import os
import sys
import pdb
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import random_split
from os.path import dirname
sys.path.append(dirname(sys.path[0]))

from data_preprocessing.MotionSense import percent

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
                                time_idx += 1000

    return


if __name__ == '__main__':
    path_save = r'./datasets/Shoaib/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    preprocess(DATASET_PATH, path_save, seq_len=200)
