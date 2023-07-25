import os
import os.path as op
import mne
import numpy as np
import pandas as pd
import scipy.io as scio


file = "original_dataset/SEED-IV/1/1_20160518.mat"
data = scio.loadmat(file)
ch_name = pd.read_excel(io='original_dataset/SEED-IV/Channel Order.xlsx')
ch_name = ch_name['FP1'].to_list()
ch_name = ['FP1'] + ch_name

for (keys, data_i) in data.items():
    if 'eeg' not in keys:
        continue
    trial = int(keys.split('_')[1][3:]) - 1
    break

sample = data_i
sfreq = 200

info = mne.create_info(ch_name, sfreq=sfreq, ch_types=['eeg' for _ in range(62)])
raw = mne.io.RawArray(sample, info)
# raw.resample(200, npad="auto")    # set sampling frequency to 256 points per second
scalings = {'eeg': 50}
fig=raw.plot(scalings=scalings)
fig.savefig("raw_eeg.png")

raw.filter(0.5, 47, fir_design='firwin', picks=ch_name)  # band-pass filter from 1 to 30 frequency over just
                                                       # EEG channel and not EEG channel
fig=raw.plot(scalings=scalings)       # plot the EEG data. Use the '%matplotlib qt' to see 
fig.savefig("filter_eeg.png")
raw.set_eeg_reference('average') # re-referencing with the virtual average reference
fig=raw.plot(scalings=scalings)       # plot the EEG data. Use the '%matplotlib qt' to see 
fig.savefig("reaverage_eeg.png")

epochs.save(examples_dir + "\\sub-006_prerprocessed.fif")