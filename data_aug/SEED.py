import os, sys
import scipy.io as scio
import numpy as np
SOURCE_DIR = 'original_dataset/SEED'
prefix = 'ww_eeg'

window_len = 200
date_list = [
    '20131027',
    '20140404',
    '20140603',
    '20140621',
    '20140411',
    '20130712',
    '20131027',
    '20140511',
    '20140620',
    '20131130',
    '20130618',
    '20131127',   
    '20140527',
    '20140601',
    '20130709',
]  # the first repetitiion is selected 

def seed_preprocess(dir_name):
    label = scio.loadmat(SOURCE_DIR + '/label.mat')
    label = label['label'][0]
    num = 0
    time_label = 0
    record = 0
    for root, dirs, files in os.walk(SOURCE_DIR):
        for f in files:
            if 'mat' not in f or 'label' in f:
                continue
            domain = int(f.split('_')[0]) - 1  # subject name
            date = f.split('_')[1]
            date = date.split('.')[0]
            if date not in date_list:
                continue
            record += 1
            
            data = scio.loadmat(SOURCE_DIR + '/' + f)
            for (keys, data_i) in data.items():
                print(keys)
                if 'eeg' not in keys:
                    continue
                trial = int(keys.split('_')[1][3:]) - 1
                data_i = data_i.transpose()
                data_i -= np.mean(data_i, axis=0)
                data_i /= np.linalg.norm(data_i, axis=0)
                
                data_i = data_i[:data_i.shape[0] // window_len * window_len, :]
                reshaped_data = data_i.reshape(-1, window_len, data_i.shape[1])
                for i in range(reshaped_data.shape[0]):
                    loc = dir_name + '/' + str(num) + '.npz'
                    np.savez(loc, eeg=reshaped_data[0], add_infor=np.array([label[trial], domain, trial, time_label]))
                    num += 1
                    time_label += 1
                
            time_label += 1000
            print(time_label)
    print(record)
    return



if __name__ == '__main__':
    dir_name = 'datasets/SEED'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    seed_preprocess(dir_name)