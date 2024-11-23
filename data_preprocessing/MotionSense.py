import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from torch.utils.data import random_split

PATH_SAVE = "./datasets/MotionSense_time/"
ORI_PATH = "./original_dataset/MotionSense/"
SEQ_LEN = 200
TARGET_WINDOW = 20
SAMPLE_WINDOW = 20
percent = [0.2, 0.5, 1, 2, 5, 10]
ACT_LABELS = ["dws", "ups", "wlk", "jog", "std", "sit"]
ACT_Translated_labels = ['Downstairs', 'Upstairs', 'Walking', 'Running', 'Standing', 'Sitting']
TRIAL_CODES = {
    ACT_LABELS[0]: [1, 2, 11],
    ACT_LABELS[1]: [3, 4, 12],
    ACT_LABELS[2]: [7, 8, 15],
    ACT_LABELS[3]: [9, 16],
    ACT_LABELS[4]: [6, 14],
    ACT_LABELS[5]: [5, 13]
}


def get_ds_infos():
    """
    Read the file includes data subject information.

    Data Columns:
    0: code [1-24]
    1: weight [kg]
    2: height [cm]
    3: age [years]
    4: gender [0:Female, 1:Male]

    Returns:
        A pandas DataFrame that contains inforamtion about data subjects' attributes 
    """

    dss = pd.read_csv("./original_dataset/MotionSense/data_subjects_info.csv")
    print("[INFO] -- Data subjects' information is imported.")

    return dss


def set_data_types(data_types=None):
    """
    Select the sensors and the mode to shape the final dataset.

    Args:
        data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration] 

    Returns:
        It returns a list of columns to use for creating time-series from files.
    """
    if data_types is None:
        data_types = ["userAcceleration"]
    dt_list = []
    for t in data_types:
        if t != "attitude":
            dt_list.append([t + ".x", t + ".y", t + ".z"])
        else:
            dt_list.append([t + ".roll", t + ".pitch", t + ".yaw"])

    return dt_list


def creat_time_series(dt_list, act_labels, trial_codes, mode="mag", labeled=True, freq=20):
    """
    Args:
        dt_list: A list of columns that shows the type of data we want.
        act_labels: list of activites
        trial_codes: list of trials
        mode: It can be "raw" which means you want raw data
        for every dimention of each data type,
        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
        or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
        labeled: True, if we want a labeld dataset. False, if we only want sensor values.

    Returns:
        It returns a time-series of sensor data.

    """
    num_data_cols = len(dt_list) if mode == "mag" else len(dt_list * 3)

    if labeled:
        dataset = np.zeros((0, num_data_cols + 7))  # "7" --> [act, code, weight, height, age, gender, trial]
    else:
        dataset = np.zeros((0, num_data_cols))

    ds_list = get_ds_infos()

    print("[INFO] -- Creating Time-Series")
    num = 0
    time_idx = 0
    for sub_id in ds_list["code"]:
        for act_id, act in enumerate(act_labels):
            for trial in trial_codes[act_id]:
                fname = ORI_PATH + 'A_DeviceMotion_data/' + act + '_' + str(trial) + '/sub_' + str(int(sub_id)) + '.csv'
                raw_data = pd.read_csv(fname)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                vals = np.zeros((len(raw_data), num_data_cols))
                for x_id, axes in enumerate(dt_list):
                    if mode == "mag":
                        vals[:, x_id] = (raw_data[axes] ** 2).sum(axis=1) ** 0.5
                    else:
                        vals[:, x_id * 3:(x_id + 1) * 3] = raw_data[axes].values
                    vals = vals[:, :num_data_cols]
                # interpolate
                acc = down_sample(vals[:, 0:3], window_target=TARGET_WINDOW)
                gyro = down_sample(vals[:, 3:6], window_target=TARGET_WINDOW)
                if len(acc) < SEQ_LEN:
                    continue
                else:
                    acc = acc[:acc.shape[0] // SEQ_LEN * SEQ_LEN, :]
                    acc = acc.reshape(-1, SEQ_LEN, acc.shape[1])
                    gyro = gyro[:gyro.shape[0] // SEQ_LEN * SEQ_LEN, :]
                    gyro = gyro.reshape(-1, SEQ_LEN, gyro.shape[1])
                    add_info = np.array([act_id, sub_id - 1]) # motion, user_id
                    for i in range(acc.shape[0]):
                        acc_new = acc[i]
                        gyro_new = gyro[i]
                        loc = PATH_SAVE + str(num) + '.npz'
                        np.savez(loc, acc=acc_new, gyro=gyro_new, add_infor=np.array([act_id, sub_id - 1, time_idx]))
                        time_idx += 1
                        num += 1
                        print(time_idx)
        time_idx += 1000
    # split_dataset(num=num-1)  # num-1 for the last void one.
    return dataset


def split_dataset(num):  # this is not cross-person setting.
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
    np.savez(os.path.join(PATH_SAVE, 'train_set' + '.npz'), train_set=train_set)
    np.savez(os.path.join(PATH_SAVE, 'val_set' + '.npz'), val_set=val_set)
    np.savez(os.path.join(PATH_SAVE, 'test_set' + '.npz'), test_set=test_set)

    loc = PATH_SAVE + 'train_set' + '.npz'
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
        loc = PATH_SAVE + 'tune_set_' + str(per).replace('.', '_') + '.npz'
        np.savez(loc, tune_set=tune_set)


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


# ________________________________


if __name__ == '__main__':
    if not os.path.exists(PATH_SAVE):
        os.makedirs(PATH_SAVE)

    sdt = ["userAcceleration", "rotationRate"]
    print("[INFO] -- Selected sensor data types: " + str(sdt))
    act_labels = ACT_LABELS[:]
    print("[INFO] -- Selected activites: " + str(act_labels))
    trial_codes = [TRIAL_CODES[act] for act in act_labels]
    dt_list = set_data_types(sdt)
    dataset = creat_time_series(dt_list, act_labels, trial_codes, mode="raw", labeled=True)
    print("[INFO] -- Shape of time-Series dataset:" + str(dataset.shape))
