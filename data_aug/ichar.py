from cProfile import label
import os
import sys
from os.path import dirname
import numpy as np
import pandas as pd
sys.path.append(dirname(sys.path[0]))

from data_aug.LPF import filter_dataset


from data_aug.preprocessing import preprocessing_dataset_cross_person_val

ACT_LABELS=['jumping', 'lying', 'running', 'sitting', 'stairdown', 'stairup',
            'standing', 'stretching', 'walking']

DEVICE_LABELS=['PH0007-jskim', 'PH0012-thanh', 'PH0014-wjlee', 'PH0034-ykha',
            'PH0038-iygoo', 'PH0041-hmkim', 'PH0045-sjlee', 'WA0002-bkkim',
            'WA0003-hskim', 'WA4697-jhryu']

FREQ_FOR_DEVICE=[100, 200, 400, 400, 500, 500, 200, 200, 100, 100]

def down_sample(data, data_freq, traget_freq):
    window_sample = data_freq * 1.0 / traget_freq
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
        while 0 <= i + window + 1 < data.shape[0]:
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


def label_name_to_index(label_names, user_names):
    label_idx = np.zeros(label_names.size)
    user_idx = np.zeros(user_names.size)
    device_idx = np.zeros(user_names.size)
    
    for i in range(len(ACT_LABELS)):
        ind = label_names == ACT_LABELS[i]
        # print(np.sum(ind))
        label_idx[ind] = i
    
    for i in range(len(DEVICE_LABELS)):
        ind = user_names == DEVICE_LABELS[i]
        user_idx[ind] = i
        
        if DEVICE_LABELS[i][0:2] == 'PH':
            device_idx[ind] = 0
        elif DEVICE_LABELS[i][0:2] == 'WA':
            device_idx[ind] = 1
        else:
            raise ValueError

    return label_idx, user_idx, device_idx



def ichar_preprocessing(path, path_save, freq_traget=20, seq_len=120):
    data = pd.read_csv(path)
    data = data.to_numpy()
    sensor = data[:, 0:6]
    activity = data[:, -2]
    subject = data[:, -1]
    subject_name = np.unique(subject)
    num=0

    activity_idx, users_idx, device_idx = label_name_to_index(activity, subject)
    for i in range(len(ACT_LABELS)):
        act_loc = activity_idx == i
        for j in range(len(DEVICE_LABELS)):
            dev_loc = users_idx == j
            loc = np.logical_and(act_loc, dev_loc)
            exp_pos = sensor[loc, :]
            exp_data = down_sample(exp_pos, FREQ_FOR_DEVICE[j] , freq_traget)
            if exp_data.shape[0] > seq_len:
                exp_data = exp_data[:exp_data.shape[0] // seq_len * seq_len, :]
                exp_data = exp_data.reshape(exp_data.shape[0] // seq_len, seq_len, exp_data.shape[1])
                exp_label = np.ones((exp_data.shape[0], exp_data.shape[1], 1))
                exp_label = np.concatenate([exp_label * device_idx[loc][0], exp_label * j, exp_label * i], 2)  # [device, users, activity]
                for m in range(exp_data.shape[0]):
                    acc_new = exp_data[m][:, 0:3]
                    gyro_new = exp_data[m][:, 3:6]
                    loc = path_save + str(num) + '.npz'
                    np.savez(loc, acc=acc_new, gyro=gyro_new, add_infor=exp_label[m])
                    num += 1
    return num-1

if __name__ == '__main__':
    path = "./original_dataset/ICHAR/ichar_original_all.csv"
    path_save = "./datasets/ICHAR_50_200/"
    num = ichar_preprocessing(path, path_save, freq_traget=50, seq_len=200)
    # filter_dataset(dir_load=path_save, dir_save=path_save, dataset_size=num, dataset_name='ICHAR')
    # preprocessing_dataset_cross_person_val(dir=path_save, dataset='ICHAR')