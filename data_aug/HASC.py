from asyncore import file_wrapper
from decimal import InvalidContext
from fileinput import filename
from http.client import MOVED_PERMANENTLY
from importlib.resources import path
import os
import numpy as np
from scipy.interpolate import interp1d
from collections import Counter


def read_meta(root, file):

    file_prefix = file.split('.')
    user = file_prefix[0]

    file_name = os.path.join(root, file)
    f = open(file_name, 'r')
    infor = f.read()
    infor = infor.split('\n')
    
    InOut=-1  # some do not have InOut infor, use -1 to indicate.

    for line in infor:
        data = line.split(':')
        if data[0] == 'TerminalType':
            device_infor = data[1].split(';')
            if len(device_infor) == 1:
                device = device_infor[0]
            else:
                device = device_infor[1]
        elif data[0] == 'Frequency(Hz)':
            freq = int(data[1])
        elif data[0] == 'Place':
            if 'outdoor' in data[1]:
                InOut = 1
            elif 'indoor' in data[1]:
                InOut = 0
            else:
                raise InvalidContext
    f.close()
    return user, device, freq, InOut


def read_movement_period(root, file):
    period = []
    file_name = os.path.join(root, file)
    try:
        f = open(file_name, 'r')
    except:
        return None
    infor = f.read()
    infor = infor.split('\n')
    for line in infor:
        if len(line) == 0 or line[0] == '#':
            continue
        data = line.split(',')
        if len(data[1]) == 0 or data[2] == ' ':
            continue
        data[0] = float(data[0])
        data[1] = float(data[1])

        if len(data[2]) == 0:
            del data[2]
        data[2] = translate_label(data[2])
        if data[2] == -1:
            continue
        period.append(data)
    f.close()
    return period

activity = ['jog', 'stdown', 'stup', 'walk', 'stay', 'jump', 'elevatordown', 'elevatorUp', ]

def translate_label(label):
    label = label.lower()

    if 'jog' in label:
        return 0
    elif 'stdown' in label or 'stairdown' in label:
        return 1
    elif 'stup' in label or 'stairup' in label:
        return 2
    elif 'walk' in label or 'move' in label:
        return 3
    elif 'skip' in label or 'jump' in label:
        return 4
    elif 'stay' in label:
        return 5
    elif 'elevatordown' in label or 'escalatordown' in label:
        return -1
    elif 'elevatorup' in label or 'escalatorup' in label:
        return -1
    elif 'terminalpositionchange' in label or 'stey' in label or 'turn' in label or 'door.maleual.close' in label:  # ignore
        return -1


def read_acc_and_gyro(root, file, period, infor, seq_len, target_freq, num, path_save):
    try:
        file_name = os.path.join(root, file+ '-acc.csv')
        data_acc=np.loadtxt(file_name, delimiter=',')
    except:
        return num


    try:
        file_name = os.path.join(root, file+ '-gyro.csv')
        data_gyro=np.loadtxt(file_name, delimiter=',')
    except:
        return num
    
    freq = infor[-1]
    for i in range(len(period)):
        sta, fin, label = period[i][0], period[i][1], period[i][2]
        add_infor = infor[0:-1]
        add_infor.append(label)

        pos_acc = np.argwhere((data_acc[:, 0] > sta-2) & (data_acc[:, 0] < fin+2)).squeeze()  # enlarge the period, to avoid interpolation error.
        pos_gyro = np.argwhere((data_gyro[:, 0] > sta-2) & (data_gyro[:, 0] < fin+2)).squeeze()
        
        extract_acc = data_acc[pos_acc, 1:]
        extract_gyro = data_gyro[pos_gyro, 1:]

        target_time = np.arange(np.max([sta, data_acc[0, 0], data_gyro[0, 0]]),
                                np.min([fin, data_acc[-1, 0], data_gyro[-1, 0]]), 
                                1/target_freq)
        
        f_acc = interp1d(data_acc[pos_acc, 0], extract_acc, axis=0, kind='linear')
        acc = f_acc(target_time) * 9.8

        f_gyro = interp1d(data_gyro[pos_gyro, 0], extract_gyro, axis=0, kind='linear')
        gyro = f_gyro(target_time)

        exp_data = np.concatenate([acc, gyro], axis=1)
        if exp_data.shape[0] > seq_len:
            exp_data = exp_data[:exp_data.shape[0] // seq_len * seq_len, :]
            exp_data = exp_data.reshape(exp_data.shape[0] // seq_len, seq_len, exp_data.shape[1])
            exp_label = np.ones((exp_data.shape[0], exp_data.shape[1], 1))
            exp_label = np.concatenate([exp_label * add_infor[0], exp_label * add_infor[1], exp_label * add_infor[2], exp_label * add_infor[3]], 2)
            for m in range(exp_data.shape[0]):
                acc_new = exp_data[m][:, 0:3]
                gyro_new = exp_data[m][:, 3:6]
                loc = path_save + str(num) + '.npz'
                np.savez(loc, acc=acc_new, gyro=gyro_new, add_infor=exp_label[m])
                num += 1
        
    return num


def main(seq_len, target_freq, path_save):
    path_original = f'./original_dataset/HASC-PAC2016/RealWorld'
    users_count = 0
    InOut_count = 0
    num = 0
    user_name = []
    device_name = []
    InOut_list = []
    freq_list = []
    label_list = []
    for root, dirs, files in os.walk(path_original):
        for f in range(len(files)):
            if 'meta' in files[f]:
                user, device, freq, InOut = read_meta(root, files[f])
                
                InOut_list.append(InOut)
                if InOut != -1:
                    InOut_count += InOut
                if device in device_name:
                    device_id = device_name.index(device)
                else:
                    device_id = len(device_name)
                    device_name.append(device)
                if user in user_name:
                    user_id = user_name.index(user)
                else:
                    user_id = len(user_name)
                    user_name.append(user)
                
                add_infor = [user_id, device_id, InOut, freq]
                freq_list.append(freq)
                period = read_movement_period(root, user + '.label')
                
                
                if period is None:
                    continue
                    
                for i in range(len(period)):
                    label_list.append(period[i][2])
                    if period[i][2] == 'movingWalkway;stay' or period[i][2] is None:
                        print('now')
                num = read_acc_and_gyro(root, user, period, add_infor, seq_len, target_freq, num, path_save)


        users_count += 1
    freq_list = np.array(freq_list)
    freq_list = np.unique(freq_list)
    label_un = np.unique(np.array(label_list))
    count = Counter(label_list)
    for i in range(len(label_un)):
        print(f'{i}: {count[i]}')

    print(label_un)
    print(freq_list)
    print(len(user_name))
    print(InOut_count)
    print(num)

    return



if __name__ == '__main__':
    path_save = f'./datasets/HASC_50_200/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    main(seq_len=200, target_freq=50, path_save=path_save)