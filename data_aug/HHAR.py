import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from collections import Counter
from torch.utils.data import random_split
import os

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

def pre_seg():
    acc = pd.read_csv('../datasets/uci_HAR/Phones_accelerometer.csv')
    gyro = pd.read_csv('../datasets/uci_HAR/Phones_gyroscope.csv')
    acc = np.asarray(acc)
    gyro = np.asarray(gyro)
    for k in users:
        pos = acc[:, -4] == k
        pos2 = gyro[:, -4] == k
        for j in devices:
            acc_user = acc[pos, :]
            gyro_user = gyro[pos2, :]
            pla = acc_user[:, -2] == j
            pla2 = gyro_user[:, -2] == j
            # if j == 'samsungold_1':
            #     print("now")
            for m in movement:
                acc_device = acc_user[pla, :]
                gyro_device = gyro_user[pla2, :]
                position = acc_device[:, -1] == m
                position2 = gyro_device[:, -1] == m
                acc_move = acc_device[position, :]
                gyro_move = gyro_device[position2, :]
                name = k + '_' + j + '_' + m
                loc = 'E:\\PyCharm 2020.3.3\\Project\\SimCLR\\datasets\\uci_HAR\\' + name + '.npz'
                np.savez(loc, acc=acc_move, gyro=gyro_move)


def after_seg():
    num = 0
    count = 0
    length = int(window / 2 / samplingRate * 1000)
    train_set = []
    val_set = []
    test_set = []
    test_users = users
    np.random.shuffle(test_users)
    test_users = test_users[:test_num_of_user]
    for k in users:
        for j in devices:
            for m in movement:
                if m == 'nan':  # nan is not considered
                    continue
                name = k + '_' + j + '_' + m
                loc = 'E:\\PyCharm 2020.3.3\\Project\\SimCLR\\datasets\\uci_HAR\\' + name + '.npz'
                data = np.load(loc, allow_pickle=True)
                acc_move = data['acc']
                gyro_move = data['gyro']
                if len(acc_move) == 0 or len(gyro_move) == 0:
                    count += 1
                    continue
                start_time = max((acc_move[0, 1], gyro_move[0, 1]))
                end_time = min(acc_move[-1, 1], gyro_move[-1, 1])
                counter_2 = 0
                low_num = num
                for i in range(start_time, end_time, length):
                    if i + window / samplingRate * 1000 + 10 > end_time:
                        break
                    t_new = np.linspace(i, i + window / samplingRate * 1000, window)
                    time = acc_move[:, 1]
                    pos = np.diff(time) != 0
                    pos = np.concatenate((pos, [True]), axis=0)
                    f_acc = interp1d(time[pos], acc_move[pos, 3:6], axis=0, kind='linear')
                    acc_new = f_acc(t_new)
                    # plt.figure_plot(time[pos], acc_window[pos, 0])
                    # plt.figure_plot(t_new, acc_new[:, 0])
                    # plt.show()

                    time = gyro_move[:, 1]
                    pos = np.diff(time) != 0
                    pos = np.concatenate((pos, [True]), axis=0)
                    f_gyro = interp1d(time[pos], gyro_move[pos, 3:6], axis=0, kind='linear')
                    gyro_new = f_gyro(t_new)

                    add_info = np.tile(acc_move[0, 6:], (len(t_new), 1))
                    loc = 'E:\\PyCharm 2020.3.3\\Project\\SimCLR\\datasets\\processed\\' + str(num) + '.npz'
                    np.savez(loc, acc=acc_new, gyro=gyro_new, add_infor=add_info)
                    counter_2 += 1
                    num += 1
                if counter_2 == 0:
                    continue
                high_num = num - 1
                if k in test_users:
                    test_num = list(range(low_num, high_num))
                    for n in test_num:
                        test_set.append(str(n) + '.npz')
                else:
                    train_set_len = int(counter_2*train_ratio)
                    train_num, val_num = random_split(range(low_num, high_num),
                                                       [train_set_len, counter_2 - 1 - train_set_len])
                    train_num = list(train_num)
                    train_num.sort()
                    val_num = list(val_num)
                    val_num.sort()

                    for n in train_num:
                        train_set.append(str(n) + '.npz')
                    for n in val_num:
                        val_set.append(str(n) + '.npz')

    #             print(k + ' ' + j + ' ' + m + ' ' + str(num))
    #         print(k + ' ' + j + ' ' + str(num))
    #     print(k + ' ' + str(num))
    # print(count)
    train_set = np.asarray(train_set)
    val_set = np.asarray(val_set)
    test_set = np.asarray(test_set)

    a = train_set
    np.random.shuffle(a)
    b = a[0:int(tune_ratio * len(a))]
    tune_set = np.sort(b)

    print(len(train_set))
    print(len(val_set))
    print(len(test_set))
    print(test_users)
    loc = 'E:\\PyCharm 2020.3.3\\Project\\SimCLR\\datasets\\processed\\' + 'train_set' + '.npz'
    np.savez(loc, train_set=train_set)
    loc = 'E:\\PyCharm 2020.3.3\\Project\\SimCLR\\datasets\\processed\\' + 'val_set' + '.npz'
    np.savez(loc, val_set=val_set)
    loc = 'E:\\PyCharm 2020.3.3\\Project\\SimCLR\\datasets\\processed\\' + 'test_set' + '.npz'
    np.savez(loc, test_set=test_set)
    loc = 'E:\\PyCharm 2020.3.3\\Project\\SimCLR\\datasets\\processed\\' + 'tune_set' + '.npz'
    np.savez(loc, tune_set=tune_set)


uot_root = 'E:\\PyCharm 2020.3.3\\Project\\SimCLR\\datasets\\uot_HAR\\'
uot_name = ['Arm.xlsx', 'Belt.xlsx', 'Wrist.xlsx', 'Pocket.xlsx']


def uot_seg():
    length = int(window / 2 / samplingRate * 1000)
    count = 0
    num = 0
    train_set = []
    val_set = []
    test_set = []

    for n in uot_name:
        loc = uot_root + n
        data = pd.read_excel(loc)
        data = np.asarray(data)
        time = data[:, 0]
        time_dif = np.diff(time)

        subject_shift = np.concatenate((np.argwhere(time_dif < 0), np.argwhere(time_dif > 5000)), axis=0)
        addition = np.ones((len(subject_shift), 1), dtype=subject_shift.dtype)
        subject_shift += addition

        subject_shift = np.concatenate(([[0]], subject_shift), axis=0)
        subject_shift = np.sort(subject_shift, axis=0)

        for k in np.arange(len(subject_shift)):
            if k == len(subject_shift) - 1:
                data_sub = data[subject_shift[k, 0]:-1, :]
            else:
                data_sub = data[subject_shift[k, 0]:subject_shift[k+1, 0], :]
            for m in uot_movement:
                position = data_sub[:, -1] == m
                acc_move = data_sub[position, 1:4]
                gyro_move = data_sub[position, 4:7]
                time = data_sub[position, 0]

                if len(acc_move) == 0 or len(gyro_move) == 0:
                    count += 1
                    continue

                start_time = time[0]
                end_time = time[-1]

                counter_2 = 0
                low_num = num
                for i in range(start_time, end_time, length):
                    if i + window / samplingRate * 1000 + 10 > end_time:
                        break
                    t_new = np.linspace(i, i + window / samplingRate * 1000, window)
                    pos = np.diff(time) != 0
                    pos = np.concatenate((pos, [True]), axis=0)
                    f_acc = interp1d(time[pos], acc_move[pos, :], axis=0, kind='linear')
                    acc_new = f_acc(t_new)

                    f_gyro = interp1d(time[pos], gyro_move[pos, :], axis=0, kind='linear')
                    gyro_new = f_gyro(t_new)

                    time_info = t_new.reshape(len(t_new), -1)
                    move_info = np.tile(m, (len(t_new), 1))
                    add_info = np.concatenate((time_info, move_info), axis=1)

                    loc = 'E:\\PyCharm 2020.3.3\\Project\\SimCLR\\datasets\\uot_processed\\' + str(num) + '.npz'
                    np.savez(loc, acc=acc_new, gyro=gyro_new, add_infor=add_info)
                    counter_2 += 1
                    num += 1
                if counter_2 == 0:
                    continue
                high_num = num - 1

                train_set_len = int(counter_2 * 0.6)
                val_set_len = int(counter_2 * 0.2)
                train_num, val_num, test_num = random_split(range(low_num, high_num),
                                                  [train_set_len, val_set_len, counter_2 - 1 - train_set_len - val_set_len])
                train_num = list(train_num)
                train_num.sort()
                val_num = list(val_num)
                val_num.sort()
                test_num = list(test_num)
                test_num.sort()

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
    loc = 'E:\\PyCharm 2020.3.3\\Project\\SimCLR\\datasets\\uot_processed\\' + 'train_set' + '.npz'
    np.savez(loc, train_set=train_set)
    loc = 'E:\\PyCharm 2020.3.3\\Project\\SimCLR\\datasets\\uot_processed\\' + 'val_set' + '.npz'
    np.savez(loc, val_set=val_set)
    loc = 'E:\\PyCharm 2020.3.3\\Project\\SimCLR\\datasets\\uot_processed\\' + 'test_set' + '.npz'
    np.savez(loc, test_set=test_set)

    a = np.concatenate((train_set, val_set))
    np.random.shuffle(a)
    tune_ratio = [1, 5, 10, 20, 50, 70, 100]
    for ratio in tune_ratio:
        b = a[0:int(ratio * 0.01 * len(a))]
        tune_set = np.sort(b)
        print(len(tune_set))
        loc = 'E:\\PyCharm 2020.3.3\\Project\\SimCLR\\datasets\\uot_processed\\' + 'tune_set_' + str(ratio) + '.npz'
        np.savez(loc, tune_set=tune_set)


def divide_fewer_labels():
    # percent = [1, 5, 10, 50, 70, 100]
    percent = [0.2, 0.5, 1, 2, 5, 10]
    loc = 'E:\\PyCharm 2020.3.3\\Project\\SimCLR\\datasets\\HHAR\\' + 'train_set' + '.npz'
    data = np.load(loc)
    train_set = data['train_set']
    # loc = 'E:\\PyCharm 2020.3.3\\Project\\SimCLR\\datasets\\processed\\' + 'val_set' + '.npz'
    # data = np.load(loc)
    # val_set = data['val_set']
    # whole_set = np.concatenate((train_set, val_set))
    whole_set = train_set
    whole_set.sort()
    np.random.shuffle(whole_set)
    set_size = len(whole_set)
    for per in percent:
        tune_set = whole_set[:int(per*0.01*set_size)]
        tune_set.sort()
        print(len(tune_set))
        loc = 'E:\\PyCharm 2020.3.3\\Project\\SimCLR\\datasets\\HHAR\\' + 'tune_set_' + str(per).replace('.', '_') + '.npz'
        np.savez(loc, tune_set=tune_set)


def verify_interpolate():
    acc = pd.read_csv('/\\datasets\\uci_HAR\\Phones_accelerometer.csv')
    acc = np.asarray(acc)
    start_time = acc[0, 1]
    stamp_acc = (start_time + window / samplingRate * 1000 - 1000 <= acc[:, 1]) & (acc[:, 1] < (start_time + 2 * window / samplingRate * 1000 + 1000))
    t_new = np.linspace(start_time + window / samplingRate * 1000, start_time + 2 * window / samplingRate * 1000, 400)
    time = acc[stamp_acc, 1]
    acc_window = acc[stamp_acc, 3:6]
    pos = np.diff(time) != 0
    pos = np.concatenate((pos, [True]), axis=0)
    f_acc = interp1d(time[pos], acc_window[pos, :], axis=0, kind='nearest')
    acc_new = f_acc(t_new)


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
            label = data_slice[['User', 'Model', 'gt']].iloc[0].values
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
                if num % 100 == 0:
                    print(num)
                # if num > 23980:
                np.savez(os.path.join(path_save, str(num) + '.npz'), acc=data_raw[:, 0:3], gyro=data_raw[:, 3:6], add_infor=data_raw[6:])
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


DATASET_PATH = r'./original_dataset/hhar/'


if __name__ == '__main__':
    # 50 refer to 20 Hz. 20 refer to 50 Hz
    path_save = r'./datasets/HHAR_50_200/'
    num = preprocess_hhar(DATASET_PATH, path_save, version='50_200', window_time=20, seq_len=200)  # use jump to control overlap.