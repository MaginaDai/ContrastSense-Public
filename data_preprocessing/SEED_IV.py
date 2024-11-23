import os
import numpy as np
import scipy.io as scio
from scipy import signal
import pandas as pd
import mne
window_len = 200

SOURCE_DIR = 'original_dataset/SEED-IV/2/'
SAMPLING_FREQ=200
# label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]  # from ReadMe
label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]

def iv_preprocess(dir_name):
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

            data = scio.loadmat(SOURCE_DIR + '/' + f)
            for (keys, data_i) in data.items():
                print(keys)
                if 'eeg' not in keys:
                    continue
                trial = int(keys.split('_')[1][3:]) - 1
                eeg_max = np.max(data_i, 1)
                eeg_min = np.min(data_i, 1)

                for i in range(62):
                    data_i[i] = (data_i[i] - eeg_min[i]) / (eeg_max[i] - eeg_min[i])

                data_i = eeg_preprocessing(data_i)
                data_i = data_i.transpose()

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


def eeg_preprocessing(sample):
    ch_name = pd.read_excel(io='original_dataset/SEED-IV/Channel Order.xlsx')
    ch_name = ch_name['FP1'].to_list()
    ch_name = ['FP1'] + ch_name

    sfreq = 200

    info = mne.create_info(ch_name, sfreq=sfreq, ch_types=['eeg' for _ in range(62)])
    raw = mne.io.RawArray(sample, info)
    raw.resample(100, npad="auto")    # set sampling frequency to 256 points per second

    raw.filter(0.5, 47, fir_design='firwin', picks=ch_name)  # band-pass filter from 1 to 30 frequency over just
                                                        # EEG channel and not EEG channel

    raw.set_eeg_reference('average') # re-referencing with the virtual average reference

    return raw['data'][0]



if __name__ == '__main__':
    dir_name = 'datasets/SEED_IV'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    iv_preprocess(dir_name)