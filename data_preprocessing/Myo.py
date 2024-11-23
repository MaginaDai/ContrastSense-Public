import numpy as np
from scipy import signal
import os

number_of_vector_per_example = 52
number_of_canals = 8
number_of_classes = 7
size_non_overlap = 52

def format_data_to_train(vector_to_format):
    dataset_example_formatted = []
    example = []
    emg_vector = []
    for value in vector_to_format:
        emg_vector.append(value)
        if (len(emg_vector) >= 8):  # sometimes we lose some channels.
            if len(example) == 0:
                example = emg_vector
            else:
                example = np.row_stack((example, emg_vector))
            emg_vector = []
            if (len(example) >= number_of_vector_per_example):
                example = example.transpose()
                dataset_example_formatted.append(example)
                example = example.transpose()
                example = example[size_non_overlap:]
    # Apply the butterworth high pass filter at 2Hz
    dataset_high_pass_filtered = []
    for example in dataset_example_formatted:
        example_filtered = []
        for channel_example in example:
            example_filtered.append(butter_highpass_filter(channel_example, 2, 200))
        dataset_high_pass_filtered.append([example_filtered])
    return np.array(dataset_high_pass_filtered)

def butter_highpass(cutoff, fs, order=3):
    nyq = .5*fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=3):
    b, a = butter_highpass(cutoff=cutoff, fs=fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def shift_electrodes(examples, labels):  # what does it do? 
    index_normal_class = [1, 2, 6, 2]  # The normal activation of the electrodes.
    class_mean = []
    # For the classes that are relatively invariant to the highest canals activation, we get on average for a
    # subject the most active canals for those classes
    for classe in range(3, 7):
        X_example = []
        Y_example = []
        for k in range(len(examples)):
            X_example.extend(examples[k])
            Y_example.extend(labels[k])

        cwt_add = []
        for j in range(len(X_example)):
            if Y_example[j] == classe:
                if len(cwt_add) == 0:
                    cwt_add = np.array(X_example[j][0])
                else:
                    cwt_add += np.array(X_example[j][0])
        class_mean.append(np.argmax(np.sum(np.array(cwt_add), axis=0)))

    # We check how many we have to shift for each channels to get back to the normal activation
    new_cwt_emplacement_left = ((np.array(class_mean) - np.array(index_normal_class)) % 10)
    new_cwt_emplacement_right = ((np.array(index_normal_class) - np.array(class_mean)) % 10)

    shifts_array = []
    for valueA, valueB in zip(new_cwt_emplacement_left, new_cwt_emplacement_right):
        if valueA < valueB:
            # We want to shift toward the left (the start of the array)
            orientation = -1
            shifts_array.append(orientation*valueA)
        else:
            # We want to shift toward the right (the end of the array)
            orientation = 1
            shifts_array.append(orientation*valueB)

    # We get the mean amount of shift and round it up to get a discrete number representing how much we have to shift
    # if we consider all the canals
    # Do the shifting only if the absolute mean is greater or equal to 0.5
    final_shifting = np.mean(np.array(shifts_array))
    if abs(final_shifting) >= 0.5:
        final_shifting = int(np.round(final_shifting))
    else:
        final_shifting = 0

    # Build the dataset of the candiate with the circular shift taken into account.
    X_example = []
    Y_example = []
    for k in range(len(examples)):
        sub_ensemble_example = []
        for example in examples[k]:
            sub_ensemble_example.append(np.roll(np.array(example), final_shifting))
        X_example.append(sub_ensemble_example)
        Y_example.append(labels[k])
    return X_example, Y_example


def read_data():
    path = 'original_dataset/Myo/PreTrainingDataset'
    print("Reading Data")
    list_dataset = []
    list_labels = []
    list_domain_labels = []

    for j, candidate in enumerate(range(12)):
        labels = []
        domain_labels = []
        examples = []
        for i in range(number_of_classes * 4):
            data_read_from_file = np.fromfile(path+'/Male'+str(candidate)+'/training0/classe_%d.dat' % i,
                                              dtype=np.int16)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
            dataset_example = format_data_to_train(data_read_from_file)
            examples.append(dataset_example)
            labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))
            domain_labels.append(j + np.zeros(dataset_example.shape[0]))

        list_dataset.append(examples)
        list_labels.append(labels)
        list_domain_labels.append(domain_labels)

    for j, candidate in enumerate(range(10)):
        labels = []
        domain_labels = []
        examples = []
        for i in range(number_of_classes * 4):
            data_read_from_file = np.fromfile(path + '/Female' + str(candidate) + '/training0/classe_%d.dat' % i,
                                              dtype=np.int16)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
            dataset_example = format_data_to_train(data_read_from_file)
            examples.append(dataset_example)
            labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))
            domain_labels.append(j + 12 + np.zeros(dataset_example.shape[0]))

        list_dataset.append(examples)
        list_labels.append(labels)
        list_domain_labels.append(domain_labels)

    path = "original_dataset/Myo/EvaluationDataset"
    for j, candidate in enumerate(range(16)):
        labels = []
        domain_labels = []
        examples = []
        for i in range(number_of_classes * 4):
            data_read_from_file = np.fromfile(path + '/Male' + str(candidate) + '/training0/classe_%d.dat' % i,
                                              dtype=np.int16)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
            dataset_example = format_data_to_train(data_read_from_file)
            examples.append(dataset_example)
            labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))
            domain_labels.append(j + 22 + np.zeros(dataset_example.shape[0]))

        list_dataset.append(examples)
        list_labels.append(labels)
        list_domain_labels.append(domain_labels)

    for j, candidate in enumerate(range(2)):
        labels = []
        domain_labels = []
        examples = []
        for i in range(number_of_classes * 4):
            data_read_from_file = np.fromfile(path + '/Female' + str(candidate) + '/training0/classe_%d.dat' % i,
                                              dtype=np.int16)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
            dataset_example = format_data_to_train(data_read_from_file)
            examples.append(dataset_example)
            labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))
            domain_labels.append(j + 38 + np.zeros(dataset_example.shape[0]))

        list_dataset.append(examples)
        list_labels.append(labels)
        list_domain_labels.append(domain_labels)
    
    print("Finished Reading Data")
    return list_dataset, list_labels, list_domain_labels


def translate_to_contrastSense_format():
    list_dataset, list_labels, list_domain_labels = read_data()
    idx = 0
    time_label = 0
    dir_name = 'datasets/Myo'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    for i, subject in enumerate(list_dataset):
        for j, motion in enumerate(subject):
            for k, sample in enumerate(motion):
                sample = np.transpose(sample, [0, 2, 1])
                label = list_labels[i][j][k]
                domain_label = list_domain_labels[i][j][k]
                file_name = f"{dir_name}/{idx}.npz"
                np.savez(file_name, emg=sample, add_infor=np.array([label, domain_label, time_label]))
                time_label += 1
                idx = idx + 1

        time_label += 1000
                

if __name__ == '__main__':
    translate_to_contrastSense_format()
