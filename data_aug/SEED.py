import os, sys
import scipy.io as scio
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

SOURCE_DIR = 'original_dataset/SEED'
prefix = 'ww_eeg'
SAMPLING_FREQ=200
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
    freq_band = np.array([4, 47])
    # wn = freq_band / SAMPLING_FREQ
    b, a = signal.butter(10, freq_band, 'bandpass', fs=SAMPLING_FREQ)

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
                ## normalization
                data_i -= np.mean(data_i, axis=0)
                data_i /= np.std(data_i, axis=0)
                ## filter
                for j in range(data_i.shape[1]):
                    # plt.figure()
                    # Y = np.fft.fft(data_i[:, j])
                    # freqs = np.fft.fftfreq(len(data_i))
                    # plt.plot(freqs[:int(Y.shape[0]/2)] * SAMPLING_FREQ, np.abs(Y)[:int(Y.shape[0]/2)], 'b')
                    
                    data_i[:, j] = signal.filtfilt(b, a, data_i[:, j])

                    # Y_after = np.fft.fft(data_i[:, j])
                    # plt.plot(freqs[:int(Y.shape[0]/2)] * SAMPLING_FREQ, np.abs(Y_after)[:int(Y.shape[0]/2)], 'r')
                    # plt.savefig('filter effect.png')
                    pass
                
                ## downsample to 100 Hz
                data_i = data_i[::2, :]
                
                data_i = data_i[:data_i.shape[0] // window_len * window_len, :]
                reshaped_data = data_i.reshape(-1, window_len, data_i.shape[1])
                for i in range(reshaped_data.shape[0]):
                    loc = dir_name + '/' + str(num) + '.npz'
                    if label[trial] == -1:
                        class_label = 2  # translate from -1 to 2, make the label as [0, 1, 2] rather than [-1, 0, 1]
                    else:
                        class_label = label[trial]
                    np.savez(loc, eeg=reshaped_data[0], add_infor=np.array([class_label, domain, trial, time_label]))
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