import wget, os
import mne
import numpy as np

sleep_stage = ['R', '1', '2', '3', '4']
SAMPLING_RATE = 100
window_len=3072

def sleep_download():
    dir_path = 'original_dataset/SleepEDF/'  # replace this with your actual directory path
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    wget.download('https://physionet.org/files/sleep-edfx/1.0.0/', dir_path)


def sleepEDFx_preprocess():
    dir_name = 'datasets/sleepEDF'
    num=0
    time_label=0

    

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    domain=0
    path = 'physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf'
    label_path = 'physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf'
    raw_data = mne.io.read_raw_edf(path, preload=True)
    raw_data = raw_data['data'][0]

    label = mne.read_annotations(label_path)
    
    
    ## preprocess
    # raw_data.set_eeg_reference('average')

    class_type = []
    for i in range(len(label)):
        class_label = label[i]['description'][-1]
        if class_label in sleep_stage:
            class_label_idx = sleep_stage.index(class_label)
        else:
            class_label_idx = -1
        class_type.append(class_label_idx)

        if class_label_idx == -1:
            continue
        
        start_idx = int(label[i]['onset'] * SAMPLING_RATE)
        duration = int(label[i]['duration'] * SAMPLING_RATE)
        
        extract_data = raw_data[0:2, start_idx: start_idx + duration]

        extract_data = extract_data.transpose()

        extract_data = extract_data[:extract_data.shape[0] // window_len * window_len, :]
        reshaped_data = extract_data.reshape(-1, window_len, extract_data.shape[1])
        for j in range(reshaped_data.shape[0]):
            loc = dir_name + '/' + str(num) + '.npz'
            np.savez(loc, eeg=reshaped_data[0], add_infor=np.array([class_label_idx, domain, time_label]))
            num += 1
            time_label += 1
    print(num)
        




    
    class_type = np.array(class_type)
    # print(np.argwhere(class_type == -1))
    class_type = np.unique(class_type)
    
    # print(class_type)
    # assign labels

    # write data

    return


if __name__ == '__main__':
    sleepEDFx_preprocess()

