from cProfile import label
from doctest import set_unittest_reportflags
from os import SCHED_RESET_ON_FORK
from xmlrpc.server import SimpleXMLRPCServer
import numpy as np
import torch
import pdb
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
from os.path import dirname
import sys
sys.path.append(dirname(sys.path[0]))
sys.path.append(dirname(dirname(sys.path[0])))
from data_preprocessing.preprocessing import UsersPosition
from exceptions.exceptions import InvalidDatasetSelection


HHAR_movement = ['stand', 'sit', 'walk', 'stairsup', 'stairsdown', 'bike']
ACT_Translated_labels = ['Downstairs', 'Upstairs', 'Walking', 'Running', 'Standing', 'Sitting']
MotionSense_Label_Seq = [4, 3, 2, 5, 0, 1]  # project from MotionSense index to UoT dataset index. Guarantee datasets classes alignment.
UCI_movement = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying']
UCI_label = [2, 3, 4, 1, 0, 0]  # we take the laying and walking as the same class. So there are 5 classes in UCI actually.
shoaib_label_Seq = [2, 1, 0, 5, 5, 3, 4]  # we take the jogging and running as the same class. So there are 6 classes in Shoaib actually.
users = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
devices = ['s3', 'nexus4', 's3mini', 'samsungold']

["walking", "sitting", "standing", "jogging", "biking", "upstairs" , "downstairs"]

class IMUDiscard(object):
    """ Discard useless information
        but keep labels
        also transform labels from str into sequence
    """

    def __init__(self, datasets_name):
        self.datasets_name = datasets_name

    def __call__(self, sample):
        acc, gyro, add_infor = sample['acc'], sample['gyro'], sample['add_infor']
        if self.datasets_name == 'HHAR' or self.datasets_name == 'HHAR filter' or self.datasets_name == 'HHAR person':
            label = np.array([HHAR_movement.index(add_infor[0, -1]), users.index(add_infor[0, -3]), devices.index(add_infor[0, -2])])
        elif self.datasets_name == 'MotionSense':
            label = np.array([ACT_Translated_labels.index(add_infor[0, -1]), int(add_infor[0, UsersPosition[self.datasets_name]])])
        elif self.datasets_name == 'UCI':
            label = np.array([int(add_infor[0, -2]), int(add_infor[0, UsersPosition[self.datasets_name]])])
        elif self.datasets_name == 'Shoaib':
            label = np.array([int(add_infor[0, -2]), int(add_infor[0, UsersPosition[self.datasets_name]])])
        else:
            raise InvalidDatasetSelection()
        return {'acc': acc, 'gyro': gyro, 'label': label}


'''
    Transformation designed by Gaole
'''


class IMUErrorModel(object):
    """ Transform the readings according to the error model

    Args:
        p: transformation possibility
        scale: the range of matrix K
        error_magn: the range of index in matrix T (except [0, 0], [1, 1], [2, 2])
        bias_magn: the range of bias

    """
    def __init__(self, p, scale, error_magn, bias_magn, pos=None):
        self.p = p
        self.scale = scale
        self.error_magn = error_magn
        self.bias_magn = bias_magn
        self.mu = 0
        self.pos = pos

    def __call__(self, sensor, pos=None):
        if (not pos and np.random.random() < self.p) or (pos and pos < self.p):
            acc, gyro = sensor[:, 0:3], sensor[:, 3:]
            K = np.eye(3)
            for i in range(3):
                K[i, i] *= np.random.uniform(low=self.scale[0], high=self.scale[1])
            K[0, 1], K[0, 2], K[1, 2] = np.random.uniform(low=-self.error_magn, high=self.error_magn, size=3)
            acc_bias = np.random.uniform(low=-self.bias_magn, high=self.bias_magn, size=3)
            acc = np.dot(acc, K) + np.tile(acc_bias, (acc.shape[0], 1))

            K = np.eye(3)
            for i in range(3):
                K[i, i] *= np.random.uniform(low=self.scale[0], high=self.scale[1])
            K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = np.random.uniform(low=-self.error_magn, high=self.error_magn, size=6)
            gyro_bias = np.random.uniform(low=-self.bias_magn, high=self.bias_magn, size=3)
            gyro = np.dot(gyro, K) + np.tile(gyro_bias, (gyro.shape[0], 1))
            sensor = np.concatenate((acc, gyro), axis=1)
        return sensor

class IMUFilter(object):
    """ filter out noise in the IMU sensors reading

    Args:
        p: transformation possibility
        cut_off_frequency: the cut-off frequency
        sampling_frequency: the sampling frequency
    """

    def __init__(self, p, cut_off_frequency, sampling_frequency, pos=None):
        self.p = p
        self.cut_off_frequency = cut_off_frequency
        self.sampling_frequency = sampling_frequency
        self.pos = pos

    def __call__(self, sample, pos=None):
        if (not pos and np.random.random() < self.p) or (pos and pos < self.p):
            acc, gyro = sample['acc'], sample['gyro']
            acc = low_pass_filter(acc, self.cut_off_frequency, self.sampling_frequency)
            gyro = low_pass_filter(gyro, self.cut_off_frequency, self.sampling_frequency)
            return {'acc': acc, 'gyro': gyro, 'label': sample['label']}
        else:
            return sample


def low_pass_filter(w, cut_off_frequency, sampling_frequency):
    N = len(w)
    w_new = np.zeros((N, w.shape[1]))
    for j in range(w.shape[1]):
        [b, a] = signal.butter(4, 2*cut_off_frequency/sampling_frequency, 'lowpass')
        w_new[:, j] = signal.filtfilt(b, a, w[:, j])
    return w_new


class IMUMalFunction(object):
    """ add artificial malfunction

    Args:
        p: transformation possibility
        mal_length: the maximum length that could be set to zero
    """

    def __init__(self, p, mal_length, pos=None):
        self.p = p
        self.mal_length = mal_length
        self.pos = pos

    def __call__(self, sensor, pos=None):
        if (not pos and np.random.random() < self.p) or (pos and pos < self.p):
            acc, gyro = sensor[:, 0:3], sensor[:, 3:]
            acc_after = np.copy(acc)
            gyro_after = np.copy(gyro)
            low = int(np.random.uniform(low=0, high=acc.shape[0] - 10))
            high = int(np.random.uniform(low=low + 2, high=min(acc.shape[0] - 1, low + self.mal_length)))
            pos = [low, high]
            f_acc = interp1d(pos, acc[pos, :], axis=0, kind='linear')
            f_gyro = interp1d(pos, gyro[pos, :], axis=0, kind='linear')
            pos_new = np.arange(low+1, high-1)
            acc_new = f_acc(pos_new)
            gyro_new = f_gyro(pos_new)

            acc_after[low+1:high-1, :] = acc_new
            gyro_after[low+1:high-1, :] = gyro_new
            sensor = np.concatenate((acc_after, gyro_after), axis=1)
        return sensor


class IMUMultiPerson(object):
    """ Prolong or shorten the whole readings

    Args:
        p: transformation possibility
        scale: prolong/shorten scale
    """

    def __init__(self, p, scale, pos=None):
        self.p = p
        self.scale = scale
        self.pos = pos

    def __call__(self, sensor, pos=None):
        # acc, gyro = sample['acc'], sample['gyro']
        if (not pos and np.random.random() < self.p) or (pos and pos < self.p):
            acc, gyro = sensor[:, 0:3], sensor[:, 3:]
            length = acc.shape[0]
            low = 0
            high = acc.shape[0]
            new_length = int(np.random.uniform(low=self.scale[0], high=self.scale[1])*length)
            pos = np.arange(low, high)
            pos_new = np.linspace(low, high - 1, new_length)

            f_acc = interp1d(pos, acc[pos, :], axis=0, kind='linear')
            f_gyro = interp1d(pos, gyro[pos, :], axis=0, kind='linear')

            acc_new = f_acc(pos_new)
            gyro_new = f_gyro(pos_new)

            if new_length >= length:
                low = int(np.random.uniform(low=0, high=new_length - length))
                acc = acc_new[low:low+length, :]
                gyro = gyro_new[low:low+length, :]
            else:
                acc_add = np.zeros((length-new_length, 3))
                gyro_add = np.zeros((length-new_length, 3))
                if np.random.random() > 0.5:  # front
                    acc = np.concatenate((acc_new, acc_add), axis=0)
                    gyro = np.concatenate((gyro_new, gyro_add), axis=0)
                else:  # back
                    acc = np.concatenate((acc_add, acc_new), axis=0)
                    gyro = np.concatenate((gyro_add, gyro_new), axis=0)
            sensor = np.concatenate((acc, gyro), axis=1)
        return sensor


'''
    Transformation designed by others
'''


class IMUNoise(object):
    """ Add noise to the IMU sensors reading

    Args:
        p: transformation possibility
        var: variance of data
    """

    def __init__(self, var, p, pos=None):
        self.p = p
        self.mu = 0
        self.var = var
        self.pos = pos

    def __call__(self, sensor, pos=None):
        if (not pos and np.random.random() < self.p) or (pos and pos < self.p):
            # acc, gyro = sample['acc'], sample['gyro']

            acc = sensor[:, 0:3]
            gyro = sensor[:, 3:]
            noise = np.random.normal(self.mu, self.var, (acc.shape[0], acc.shape[1]))
            acc += noise

            noise = np.random.normal(self.mu, self.var, (gyro.shape[0], gyro.shape[1]))
            gyro += noise
            sensor = np.concatenate((acc, gyro), axis=1)

        return sensor


class IMUScale(object):
    """ Rescale the IMU sensors reading

    Args:
        scale: the scaling range for all axis
        p: transformation possibility
    """

    def __init__(self, scale, p, pos=None):
        self.scale = scale
        self.p = p
        self.pos = pos

    def __call__(self, sensor, pos=None):
        if (not pos and np.random.random() < self.p) or (pos and pos < self.p):
            # acc, gyro = sample['acc'], sample['gyro']
            acc, gyro = sensor[:, 0:3], sensor[:, 3:]
            K = np.eye(3)
            for i in range(3):
                K[i, i] *= np.random.uniform(low=self.scale[0], high=self.scale[1])
            acc = np.dot(acc, K)

            K = np.eye(3)
            for i in range(3):
                K[i, i] *= np.random.uniform(low=self.scale[0], high=self.scale[1])
            gyro = np.dot(gyro, K)
            sensor = np.concatenate((acc, gyro), axis=1)
        return sensor


class IMURotate(object):
    """ Rotate the IMU sensors reading

    Args:
        p: transformation possibility
    """

    def __init__(self, p, pos=None):
        self.p = p
        self.pos = pos

    def __call__(self, sensor, pos=None):
        if (not pos and np.random.random() < self.p) or (pos and pos < self.p):
            acc, gyro = sensor[:, 0:3], sensor[:, 3:]

            axis_x = np.random.random((1, 3)) * 2 - 1
            axis_y = np.random.random((1, 3)) * 2 - 1

            axis_x /= np.linalg.norm(axis_x)
            axis_y /= np.linalg.norm(axis_y)

            axis_z = np.cross(axis_x, axis_y)
            axis_y = np.cross(axis_z, axis_x)

            R = np.concatenate([axis_x, axis_y, axis_z], axis=0)
            acc = np.dot(acc, R)
            gyro = np.dot(gyro, R)
            sensor = np.concatenate((acc, gyro), axis=1)
        return sensor


class IMUNegated(object):
    """ Negate the IMU sensors reading

    Args:
        p: transformation possibility
    """

    def __init__(self, p, pos=None):
        self.p = p
        self.pos = pos

    def __call__(self, sensor, pos=None):
        if (not pos and np.random.random() < self.p) or (pos and pos < self.p):
            acc, gyro = sensor[:, 0:3], sensor[:, 3:]
            R = np.eye(3)
            for i in range(3):
                if np.random.random() < 0.5:
                    R[i, i] = -1
            acc = np.dot(acc, R)
            gyro = np.dot(gyro, R)
            sensor = np.concatenate((acc, gyro), axis=1)
        return sensor


class IMUHorizontalFlip(object):
    """ Flip the IMU sensors reading

    Args:
        p: transformation possibility
    """

    def __init__(self, p, pos=None):
        self.p = p
        self.pos = pos

    def __call__(self, sensor, pos=None):
        if (not pos and np.random.random() < self.p) or (pos and pos < self.p):
            acc, gyro = sensor[:, 0:3], sensor[:, 3:]
            acc = acc[::-1, :]
            gyro = gyro[::-1, :]
            sensor = np.concatenate((acc, gyro), axis=1)
        return sensor


class IMUTimeWarp(object):
    """ locally prolong the IMU sensors reading

    Args:
        p: transformation possibility
    """

    def __init__(self, p, pos=None):
        self.p = p
        self.pos = pos

    def __call__(self, sensor, pos=None):
        if (not pos and np.random.random() < self.p) or (pos and pos < self.p):
            acc, gyro = sensor[:, 0:3], sensor[:, 3:]
            length = acc.shape[0]
            low = int(np.random.uniform(low=0, high=acc.shape[0]-10))
            high = int(np.random.uniform(low=low+2, high=acc.shape[0]-5))
            new_high = int(np.random.uniform(low=high+3, high=acc.shape[0]-1))
            new_length = new_high - low
            pos = np.arange(low, high)
            pos_new = np.linspace(low, high-1, new_length)
            f_acc = interp1d(pos, acc[pos, :], axis=0, kind='linear')
            f_gyro = interp1d(pos, gyro[pos, :], axis=0, kind='linear')
            acc_new = f_acc(pos_new)
            gyro_new = f_gyro(pos_new)
            acc_scale = np.concatenate((acc[:low, :], acc_new, acc[high:]), axis=0)
            gyro_scale = np.concatenate((gyro[:low, :], gyro_new, gyro[high:]), axis=0)
            sensor = np.concatenate((acc_scale[:length, :], gyro_scale[:length, :]), axis=1)
        return sensor


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sensor):
        acc, gyro = sensor[:, 0:3], sensor[:, 3:]
        acc = acc/9.8   # normalization
        sensor = np.concatenate((acc, gyro), axis=1).astype('float32')
        return torch.from_numpy(sensor.reshape(-1, sensor.shape[0], sensor.shape[1]))


class IMU_Resampling(object):
    """ https://arxiv.org/abs/2109.02054 """
    def __init__(self, M, N):
        self.M = M
        self.N = N

    def __call__(self, sensor):
        total_len = self.M * (sensor.shape[0] - 1) +  sensor.shape[0]
        s0 = sensor[:-1, :]
        s1 = sensor[1:, :]
        
        dif = (s1 - s0) / (self.M+1)
        sensor_after = np.zeros([self.M, sensor.shape[0] - 1, sensor.shape[1]])
        for i in range(self.M):
            sensor_after[i] = s0 + dif * (i+1)
        s2 = np.zeros([total_len, sensor.shape[1]])
        step = np.arange(0, total_len, self.M+1)
        s2[step, :] = sensor
        for i in range(self.M):
            step = np.arange(i+1, total_len, self.M+1)
            s2[step, :] = sensor_after[i]
        start = np.random.randint(0, total_len - sensor.shape[0] * self.N)
        step = np.arange(start, total_len, self.N)
        return s2[step[:sensor.shape[0]], :]

if __name__ is '__main__':
    sensor = np.random.randn(4, 3)
    resample = IMU_Resampling(M=2, N=2)
    s2 = resample(sensor, 2, 2)
    print(s2)
