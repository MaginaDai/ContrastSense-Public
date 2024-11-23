import os
import sys
from os.path import dirname
import numpy as np
from torch.utils.data import random_split

sys.path.append(dirname(sys.path[0]))
from data_preprocessing.MotionSense import percent

DATASET_PATH = r'./original_dataset/uci/RawData'


def down_sample(data, window_sample, start, end):
    result = []
    if window_sample.is_integer():
        window = int(window_sample)
        for i in range(start, end - window, window):
            slice = data[i: i + window, :]
            result.append(np.mean(slice, 0))
    else:
        window = int(window_sample)
        remainder = 0.0
        i = int(start)
        while int(start) <= i + window + 1 < int(end):
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


def preprocess(path, path_save, raw_sr=50, target_sr=50, seq_len=200):
    labels = np.loadtxt(os.path.join(DATASET_PATH, 'labels.txt'), delimiter=' ')
    window_sample = raw_sr / target_sr
    num = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.startswith('acc'):
                tags = name.split('.')[0].split('_')
                exp_num = int(tags[1][-2:])
                exp_user = int(tags[2][-2:])
                label_index = (labels[:, 0] == exp_num) & (labels[:, 1] == exp_user)
                label_stat = labels[label_index, :]
                for i in range(label_stat.shape[0]):
                    index_start = label_stat[i, 3]
                    index_end = label_stat[i, 4]
                    exp_data_acc = np.loadtxt(os.path.join(root, name), delimiter=' ') * 9.80665
                    exp_data_gyro = np.loadtxt(os.path.join(root, 'gyro' + name[3:]), delimiter=' ')
                    exp_data = down_sample(np.concatenate([exp_data_acc, exp_data_gyro], 1), window_sample, index_start,
                                           index_end)
                    if exp_data.shape[0] > seq_len and label_stat[i, 2] <= 6:
                        exp_data = exp_data[:exp_data.shape[0] // seq_len * seq_len, :]
                        exp_data = exp_data.reshape(exp_data.shape[0] // seq_len, seq_len, exp_data.shape[1])
                        exp_label = np.ones((exp_data.shape[0], exp_data.shape[1], 1))
                        exp_label = np.concatenate([exp_label * (label_stat[i, 1] - 1), exp_label * (label_stat[i, 2] - 1), exp_label * (exp_user)], 2)  # -1 to make it start from 0
                        for m in range(exp_data.shape[0]):
                            acc_new = exp_data[m][:, 0:3]
                            gyro_new = exp_data[m][:, 3:6]
                            loc = path_save + str(num) + '.npz'
                            np.savez(loc, acc=acc_new, gyro=gyro_new, add_infor=exp_label[m])
                            num += 1
    # split_dataset(num=num - 1)  # num-1 for the last void one.
    print(num-1)
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


# activity, user
path_save = r'./datasets/UCI_50_200/'
# split_dataset(num=2087)  # num-1 for the last void one.
preprocess(DATASET_PATH, path_save, target_sr=20, seq_len=200)
