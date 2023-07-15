import os
import numpy as np
import scipy.io as scio
from scipy import signal
window_len = 200

SOURCE_DIR = 'original_dataset/SEED-IV/1/'
SAMPLING_FREQ=200
label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]  # from ReadMe

def iv_preprocess(dir_name):
    num = 0
    time_label = 0
    record = 0
    freq_band = np.array([4, 47])
    b, a = signal.butter(10, freq_band, 'bandpass', fs=SAMPLING_FREQ)

    for root, dirs, files in os.walk(SOURCE_DIR):
        for f in files:
            if 'mat' not in f or 'label' in f:
                continue
            domain = int(f.split('_')[0]) - 1  # subject name
            date = f.split('_')[1]
            date = date.split('.')[0]

            data = scio.loadmat(SOURCE_DIR + '/' + f)
            for (keys, data_i) in data.items():
                print(keys)
                if 'eeg' not in keys:
                    continue
                trial = int(keys.split('_')[1][3:]) - 1
                data_i = data_i.transpose()
                data_i -= np.mean(data_i, axis=0)
                data_i /= np.std(data_i, axis=0)
                for j in range(data_i.shape[1]):
                    data_i[:, j] = signal.filtfilt(b, a, data_i[:, j])

                ## downsample to 100 Hz
                data_i = data_i[::2, :]

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


if __name__ == '__main__':
    dir_name = 'datasets/SEED_IV'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    iv_preprocess(dir_name)