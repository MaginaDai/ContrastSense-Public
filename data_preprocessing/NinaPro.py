import zipfile
import scipy.io
import numpy as np
import os


subject_num = 10
channel_subset = (8, 15)  ## lower arm

feature_names = [[
    "ch1", "ch2", "ch3", "ch4", "ch5",
    "ch6", "ch7", "ch8",
]]
repetitions = [1, 2, 3, 4, 5, 6]

window_size = 52  # 200Hz * 0.260s (260ms) = 52 samples
window_overlap = False # which means no overlap
users = 10
which_ex=2
label_subset=[
            0,   # Neutral
            15,  # "Wrist radial deviation"
            13,  # "Wrist flexion"
            16,  # "Wrist ulnar deviation"
            14,  # "Wrist extension"
            6,   # "Fingers flexed together in fist" (hand closed?)
            5,   # "Abduction of all fingers" (hand open?)
]
shift_electrodes=False


num_classes = 7
class_labels = [
    "Neutral",
    "RadialDeviation",
    "WristFlexion",
    "UlnarDeviation",
    "WristExtension",
    "HandClose",
    "HandOpen",
]

def get_data():
    dir_name = './datasets/NinaPro_time/'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    data = []
    label = []
    num = 0
    time_label = 0
    for i in range(10):
        data_i, label_i = get_data_for_one_user(i+1)
        data_i = np.expand_dims(data_i, axis=1)
        domain_label_i = i * np.ones(len(label_i))
        label_i = np.vstack([label_i, domain_label_i])
    
        data.append(data_i)
        label.append(label_i)

        for j in range(len(data_i)):
            np.savez(dir_name + str(num) + '.npz', emg = data_i[j], add_infor=np.array([label_i[0, j], label_i[1, j], time_label]))
            num+=1
            time_label+=1
            print(time_label)
        
        time_label += 1000
        

    # data = np.vstack(data)
    # label = np.hstack(label)

    # for i in range(len(data)):
    #     np.savez('./datasets/NinaPro/' + str(i) + '.npz', emg = data[i], add_infor=label[:, i])
    return
        
    

def get_data_for_one_user(user_num, if_cda=False):
    fp = f'original_dataset/NinaPro/s{user_num}/S{user_num}_E2_A1.mat'
    mat = scipy.io.loadmat(fp)
    xs = mat["emg"]

    # Channel subset
    if channel_subset is not None:
        assert isinstance(channel_subset, tuple) \
            and len(channel_subset) == 2, \
            "channel_subset should be of the form (start_ch, end_ch)"

        channel_start, channel_end = channel_subset
        # Select start to end, inclusive, thus end+1
        xs = xs[:, channel_start:channel_end+1]
        ys = mat["restimulus"]
        reps = mat["rerepetition"]
        
        label_set = np.unique(mat["restimulus"])

    # Load the x/y data for the desired set of repetitions
    data = []
    labels = []
    windows_per_label = []

    # We handle rest differently because otherwise there's way more "rest"
    # than the other classes
    data_rest = []
    labels_rest = []

    if label_subset is not None:
        assert isinstance(label_subset, list), \
            "label_subset should be a list"
        assert all([
            label in label_set for label in label_subset
        ]), "not all labels in label_subset are found in the file"

        all_classes = label_subset
    else:
        all_classes = list(range(num_classes))

    
    for label_index, label in enumerate(all_classes):
        windows_per_this_label = 0

        for rep in repetitions:
            # Get just the data with this label for this repetition
            #
            # Note: we skip rest, so label 0 is the first movement not rest,
            # if not include_rest.
            wh = np.where(np.squeeze(np.logical_and(
                ys == label, reps == rep), axis=1))
            x = xs[wh]

            x = create_windows_x(x, window_size, window_overlap)
            y = label_index * np.ones(len(x))

            if if_cda:
                if x.shape[0] % 2 == 1:
                    x = x[:-1]
                    y = y[:-1]
                x = x.reshape([int(x.shape[0]/2), 2, x.shape[1], x.shape[2]])
                y = y.reshape([int(y.shape[0]/2), 2])

            windows_per_this_label += len(x)

            if label == 0:
                data_rest.append(x)
                labels_rest.append(y)
            else:
                data.append(x)
                labels.append(y)
            
        if label != 0:
            windows_per_label.append(windows_per_this_label)

    avg_windows_others = np.mean(windows_per_label).astype(np.int32)
    # print("Taking only first", avg_windows_others, "of rest")

    # Concat all labels/reps, otherwise if we take the subset first we
    # essentially get all the data
    data_rest = np.vstack(data_rest).astype(np.float32)
    if if_cda:
        labels_rest = np.vstack(labels_rest).astype(np.float32)
    else:
        labels_rest = np.hstack(labels_rest).astype(np.float32)

    # Shuffle both together; we don't want to always get just the first
    # "rest" instances. Also, make this repeatable.
    p = np.random.RandomState(seed=940).permutation(len(data_rest))
    data_rest = data_rest[p]
    labels_rest = labels_rest[p]

    # Limit the number, also put at the beginning in case
    # shift_electrodes
    data.append(data_rest[:avg_windows_others])
    labels.append(labels_rest[:avg_windows_others])

    data = np.vstack(data).astype(np.float32)

    if if_cda:
        labels = np.vstack(labels).astype(np.float32)
    else:
        labels = np.hstack(labels).astype(np.float32)
    return data, labels

def create_windows_x(x, window_size, overlap):
    """
    Concatenate along dim-1 to meet the desired window_size. We'll skip any
    windows that reach beyond the end. Only process x (saves memory).

    Three options (examples for window_size=5):
        Overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
            label of example 4; and window 1 will be 1,2,3,4,5 and the label of
            example 5
        No overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
            label of example 4; and window 1 will be 5,6,7,8,9 and the label of
            example 9
        Overlap as integer rather than True/False - e.g. if overlap=2 then
            window 0 will be examples 0,1,2,3,4 and then window 1 will be
            2,3,4,5,6, etc.
    """
    x = np.expand_dims(x, axis=1)

    # No work required if the window size is 1, only part required is
    # the above expand dims
    if window_size == 1:
        return x

    windows_x = []
    i = 0

    while i < len(x)-window_size:
        window_x = np.expand_dims(np.concatenate(x[i:i+window_size], axis=0), axis=0)
        windows_x.append(window_x)
        # Where to start the next window
        i = window_next_i(i, overlap, window_size)

    return np.vstack(windows_x)

def window_next_i(i, overlap, window_size):
    """ Where to start the next window """
    if overlap is not False:
        if overlap is True:
            i += 1
        elif isinstance(overlap, int):
            i += overlap
        else:
            raise NotImplementedError("overlap should be True/False or integer")
    else:
        i += window_size

    return i

def get_data_for_CDA():
    data = []
    label = []
    
    for i in range(10):
        data_i, label_i = get_data_for_one_user(i+1, if_cda=True)
        data_i = np.expand_dims(data_i, axis=2)
        domain_label_i = np.expand_dims(i * np.ones([label_i.shape[0], label_i.shape[1]]), axis=2)
        label_i = np.expand_dims(label_i, axis=2)
        label_i = np.concatenate([label_i, domain_label_i], axis=2)
    
        data.append(data_i)
        label.append(label_i)

    data = np.vstack(data)
    label = np.vstack(label)

    for i in range(len(data)):
        np.savez('./datasets/NinaPro_cda/' + str(i) + '.npz', emg = data[i], add_infor=label[i, 0])  # since the label for the two are always the same, we just keep one of them, to make the split more easier.


if __name__ == '__main__':
    # get_data_for_CDA()
    get_data()