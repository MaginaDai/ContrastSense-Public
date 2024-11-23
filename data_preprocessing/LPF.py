from fileinput import filename
import os
from os.path import dirname
import sys
from tkinter.tix import DirSelectBox
import numpy as np
import torch
from scipy.interpolate import interp1d

# from data_loader.preprocessing import movement, uot_movement
from scipy import signal



def low_pass_filter(w, cut_off_frequency, sampling_frequency):
    N = len(w)
    w_new = np.zeros((N, w.shape[1]))
    for j in range(w.shape[1]):
        [b, a] = signal.butter(4, cut_off_frequency/sampling_frequency, 'lowpass')
        w_new[:, j] = signal.filtfilt(b, a, w[:, j])
    return w_new


def filter_dataset(dir_load, dir_save, dataset_size, sampling_frequency):
    print("start filtering")
    for i in range(dataset_size):
        sub_dir = dir_load + str(i) + '.npz'
        sto_dir = dir_save + str(i) + '.npz'
        data = np.load(sub_dir, allow_pickle=True)
        acc = low_pass_filter(data['acc'], cut_off_frequency=12, sampling_frequency=sampling_frequency)
        gyro = low_pass_filter(data['gyro'], cut_off_frequency=12, sampling_frequency=sampling_frequency)
        np.savez(sto_dir, acc=acc, gyro=gyro, add_infor=data['add_infor'])
    print("end filtering")

if __name__ == '__main__':
    dir_save=r'./datasets/ICHAR person/'
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    filter_dataset(dir_load=r'./datasets/ICHAR/', dir_save=dir_save, dataset_name='ICHAR')
    
    




