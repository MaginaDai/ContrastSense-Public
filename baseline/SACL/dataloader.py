import torch
import numpy as np
import pickle as pkl
import random, os
from scipy.signal import iirnotch, lfilter


class SADataset(torch.utils.data.Dataset): # NOTE THAT IF perturb_orig_signal==False AND individual_id is None AND num_individuals_in_label is None, THEN THIS DATASET *SHOULD* BE EQUIVALENT TO SQDataset
    def __init__(self, cached_ps_dataset, path=None, total_points=None, window_size=None, sfreq=1000, bw=5, 
                 randomized_augmentation=False, num_channels=11, temporal_len=3000, NUM_AUGMENTATIONS=2, perturb_orig_signal=True, individual_id=None, num_individuals_in_label=None, 
                 windowed_data_name="_Windowed_Pretext_Preprocess.npy", windowed_start_time_name="_Windowed_StartTime.npy"):
        if cached_ps_dataset is not None:
            self.init_from_cached_data(cached_ps_dataset)
        else:
            self.init_params_from_scratch(path, 
                                          total_points, 
                                          window_size, 
                                          sfreq, 
                                          bw, 
                                          randomized_augmentation, 
                                          num_channels, 
                                          temporal_len, 
                                          NUM_AUGMENTATIONS, 
                                          perturb_orig_signal, 
                                          individual_id, 
                                          num_individuals_in_label, 
                                          windowed_data_name, 
                                          windowed_start_time_name
            )
        pass 

    def init_params_from_scratch(self, path, total_points, window_size, sfreq, bw, 
                                 randomized_augmentation, num_channels, temporal_len, 
                                 NUM_AUGMENTATIONS,  perturb_orig_signal, individual_id, num_individuals_in_label, 
                                 windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                                 windowed_start_time_name="_Windowed_StartTime.npy"):
        self.available_augmentations = {
            "amplitude_scale": [0.5, 2], 
            "time_shift": [-50, 50], 
            "DC_shift": [-10, 10], 
            "zero-masking": [0, 150], 
            "additive_Gaussian_noise": [0, 0.2], 
            "band-stop_filter": [2.8, 82.5], 
        }
        self.TEMPORAL_DIM = 0
        self.CHANNEL_DIM = 1
        self.NUM_AUGMENTATIONS = NUM_AUGMENTATIONS
        self.perturb_orig_signal = perturb_orig_signal
        self.individual_id = individual_id
        self.num_individuals_in_label = num_individuals_in_label
        self.NUM_CHANNELS = num_channels
        self.TEMPORAL_LEN = temporal_len
        self.SFREQ = sfreq
        self.BW = bw # band width (?) see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html

        data_path = path + windowed_data_name
        self.data = np.load(data_path)
        # print("SADataset.init_params_from_scratch(): data size == ", self.data.size)
        # print("SADataset.init_params_from_scratch(): data shape == ", self.data.shape)

        data_path = path + windowed_start_time_name
        self.start_times = np.load(data_path)
        self.total_windows = len(self.data)
        self.randomized_augmentation = randomized_augmentation
        self.pairs, self.labels = self.get_samples_and_labels(size=total_points, window_size=window_size)
        pass
    
    def init_from_cached_data(self, cached_rp_dataset):
        cached_dataset = None
        with open(cached_rp_dataset, 'rb') as infile:
            cached_dataset = pkl.load(infile)
        
        self.available_augmentations = cached_dataset['available_augmentations']
        self.TEMPORAL_DIM = cached_dataset['TEMPORAL_DIM']
        self.CHANNEL_DIM = cached_dataset['CHANNEL_DIM']
        self.NUM_AUGMENTATIONS = cached_dataset['NUM_AUGMENTATIONS']
        self.perturb_orig_signal = cached_dataset['perturb_orig_signal']
        self.individual_id = cached_dataset['individual_id']
        self.num_individuals_in_label = cached_dataset['num_individuals_in_label']
        self.NUM_CHANNELS = cached_dataset['NUM_CHANNELS']
        self.TEMPORAL_LEN = cached_dataset['TEMPORAL_LEN']
        self.SFREQ = cached_dataset['SFREQ']
        self.BW = cached_dataset['BW']
        self.data = cached_dataset['data']
        # print("SADataset.init_from_cached_data(): data size == ", self.data.size)
        # print("SADataset.init_from_cached_data(): data shape == ", self.data.shape)
        self.start_times = cached_dataset['start_times']
        self.total_windows = cached_dataset['total_windows']
        self.randomized_augmentation = cached_dataset['randomized_augmentation']
        self.pairs = cached_dataset['pairs']
        self.labels = cached_dataset['labels']

        del cached_dataset
        pass

    def save_as_dictionary(self, path):
        with open(path, 'wb') as outfile:
            pkl.dump({
                'available_augmentations': self.available_augmentations, 
                'TEMPORAL_DIM': self.TEMPORAL_DIM, 
                'CHANNEL_DIM': self.CHANNEL_DIM, 
                'NUM_AUGMENTATIONS': self.NUM_AUGMENTATIONS, 
                'perturb_orig_signal': self.perturb_orig_signal, 
                'individual_id': self.individual_id, 
                'num_individuals_in_label': self.num_individuals_in_label, 
                'NUM_CHANNELS': self.NUM_CHANNELS, 
                'TEMPORAL_LEN': self.TEMPORAL_LEN, 
                'SFREQ': self.SFREQ, 
                'BW': self.BW, 
                'data': self.data,
                'start_times': self.start_times,
                'total_windows': self.total_windows,
                'randomized_augmentation': self.randomized_augmentation, 
                'pairs': self.pairs,
                'labels': self.labels,
            }, outfile)

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        x_t1 = self.data[self.pairs[index][0], :, :]
        x_t2 = self.data[self.pairs[index][0], :, :]

        # APPLY RELEVANT TRANSFORMATION T1 TO X_T1
        if self.perturb_orig_signal:
            if self.pairs[index][1][0] is not None: 
                # print("SADataset.__getitem__: using STORED augmentations set self.pairs[index][1][0] == ", self.pairs[index][1][0])
                x_t1 = self.apply_augmentations(x_t1, self.pairs[index][1][0])
            else:
                # print("SADataset.__getitem__: using NEW augmentations set because self.pairs[index][1][0] == ", self.pairs[index][1][0])
                curr_augmentations = self.get_augmentation_set()
                # print("SADataset.__getitem__: using NEW augmentations set curr_augmentations == ", curr_augmentations)
                x_t1 = self.apply_augmentations(x_t1, curr_augmentations)

        # APPLY RELEVANT TRANSFORMATION T2 TO X_T2
        if self.pairs[index][1][1] is not None: 
            # print("SADataset.__getitem__: using STORED augmentations set self.pairs[index][1][1] == ", self.pairs[index][1][1])
            x_t2 = self.apply_augmentations(x_t2, self.pairs[index][1][1])
        else:
            # print("SADataset.__getitem__: using NEW augmentations set because self.pairs[index][1][1] == ", self.pairs[index][1][1])
            curr_augmentations = self.get_augmentation_set()
            # print("SADataset.__getitem__: using NEW augmentations set curr_augmentations == ", curr_augmentations)
            x_t2 = self.apply_augmentations(x_t2, curr_augmentations)

        x_t1 = torch.from_numpy(x_t1).float()
        x_t2 = torch.from_numpy(x_t2).float()

        # y = np.zeros(num_individuals_in_label)
        # y[self.individual_id] = 1.
        y = torch.from_numpy(self.labels).float()

        return x_t1, x_t2, y
    
    def get_samples_and_labels(self, size, window_size):
        """
        Gets the pairs of inputs [x, [t1, t2]] and output labels for the pretext task. 

        Each pair, representing the signal to be transformed by randomly chosen t1 and t2, is given the same label, 
            corresponding to the test subject from which the signals were taken.

        see Sections 3.1 and 3.2 of arxiv.org/pdf/2007.04871.pdf
        """
        pairs = [[None, None] for _ in range(size)] # np.zeros((size, 2), dtype=int)
        labels = np.zeros(self.num_individuals_in_label)
        labels[self.individual_id] = 1.

        for i in range(size):
            anchor_val = np.random.randint(low=0, high=self.total_windows)
            second_val = [None, None]
            if not self.randomized_augmentation: # decide whether or not to save augmentation strategy for curr sample
                if self.perturb_orig_signal:
                    second_val[0] = self.get_augmentation_set()
                second_val[1] = self.get_augmentation_set()
            # else:
            #     second_val = [None, None]

            pairs[i][0] = anchor_val
            pairs[i][1] = second_val

        # print("SADataset.get_samples_and_labels: labels shape == ", labels.shape)
        return pairs, labels
    
    def get_augmentation_set(self):
        """
        see Section 2.3 of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf
        """
        augmentation_set = [] # [dict()]*self.NUM_CHANNELS
        
        for j in range(self.NUM_CHANNELS):
            augmentation_set.append(dict())
            selected_augmentations = random.sample(list(self.available_augmentations.keys()), self.NUM_AUGMENTATIONS) # see https://pynative.com/python-random-sample/#:~:text=Python's%20random%20module%20provides%20random,it%20random%20sampling%20without%20replacement.
            assert len(selected_augmentations) == self.NUM_AUGMENTATIONS
            # print("SADataset.get_augmentation_set: selected_augmentations == ", selected_augmentations)
            # print("SADataset.get_augmentation_set: len(selected_augmentations) == 2")
            counter = 0
            for _, curr_augmentation in enumerate(selected_augmentations):
                curr_augmentation_val = None

                if curr_augmentation in ['amplitude_scale', 'DC_shift', 'additive_Gaussian_noise', 'band-stop_filter']: # augmentation that requires float val
                    curr_augmentation_val = random.uniform(self.available_augmentations[curr_augmentation][0], self.available_augmentations[curr_augmentation][1]) # see https://stackoverflow.com/questions/6088077/how-to-get-a-random-number-between-a-float-range

                elif curr_augmentation in ['time_shift', 'zero-masking']: # augmentation that requires int val
                    curr_augmentation_val = random.randint(self.available_augmentations[curr_augmentation][0], self.available_augmentations[curr_augmentation][1]) # see https://stackoverflow.com/questions/3996904/generate-random-integers-between-0-and-9
                    if curr_augmentation == 'zero-masking':
                        curr_augmentation_val = [curr_augmentation_val, random.randint(0, self.TEMPORAL_LEN-1)]

                else:
                    raise NotImplementedError("curr_augmentation == "+str(curr_augmentation)+" not recognized for value sampling")

                augmentation_set[j][curr_augmentation] = curr_augmentation_val
                counter += 1
            # print("SADataset.get_augmentation_set: augmentation_set == ", augmentation_set)
            # print("SADataset.get_augmentation_set: augmentation_set[j].keys() == ", augmentation_set[j].keys())
            assert len(list(augmentation_set[j].keys())) == self.NUM_AUGMENTATIONS
            assert counter == self.NUM_AUGMENTATIONS
        # raise NotImplementedError()
        return augmentation_set
    
    def apply_augmentations(self, x, augmentations):
        """
        see Section 2.2 of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf
        """
        # print("\n#####################\nSADataset.apply_augmentations: x shape == ", x.shape)
        if augmentations is not None:
            # print("SADataset.apply_augmentations: augmentations == ", augmentations)
            assert len(augmentations) == 11
            for j, curr_augmentation_set in enumerate(augmentations):
                assert len(list(curr_augmentation_set.keys())) == 2
                # print("SADataset.apply_augmentations: len(list(curr_augmentation_set.keys())) == 2")
                # print("\tSADataset.apply_augmentations: curr_augmentation_set == ", curr_augmentation_set)
                for _, curr_augmentation in enumerate(list(curr_augmentation_set.keys())):
                    # print("\t\tSADataset.apply_augmentations: curr_augmentation == ", curr_augmentation)
                    curr_augmentation_val = curr_augmentation_set[curr_augmentation]

                    if curr_augmentation == 'amplitude_scale':
                        # print("\n\t\t\tSADataset.apply_augmentations: NOW APPLYING amplitude_scale AUGMENTATION")
                        # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                        x[:,j] = curr_augmentation_val * x[:,j]
                        # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                    elif curr_augmentation == 'DC_shift':
                        # print("\n\t\t\tSADataset.apply_augmentations: NOW APPLYING DC_shift AUGMENTATION")
                        # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                        x[:,j] = x[:,j] + curr_augmentation_val
                        # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                    elif curr_augmentation == 'additive_Gaussian_noise':
                        # print("\n\t\t\tSADataset.apply_augmentations: NOW APPLYING additive_Gaussian_noise AUGMENTATION")
                        # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                        # print("\t\t\tSADataset.apply_augmentations: np.random.normal(0, curr_augmentation_val, x[:,j].shape) shape == ", np.random.normal(0, curr_augmentation_val, x[:,j].shape).shape)
                        x[:,j] = x[:,j] + np.random.normal(0, curr_augmentation_val, x[:,j].shape)# see https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python and https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
                        # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                    elif curr_augmentation == 'band-stop_filter':
                        """
                        see:
                            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html
                            https://www.programcreek.com/python/example/115815/scipy.signal.iirnotch
                            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
                        """
                        # print("\n\t\t\tSADataset.apply_augmentations: NOW APPLYING band-stop_filter AUGMENTATION")
                        # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                        # print("\t\t\tSADataset.apply_augmentations: curr_augmentation_val == ", curr_augmentation_val)
                        # print("\t\t\tSADataset.apply_augmentations: curr_augmentation_val/self.BW == ", curr_augmentation_val/self.BW)
                        # print("\t\t\tSADataset.apply_augmentations: self.SFREQ == ", self.SFREQ)
                        b, a = iirnotch(curr_augmentation_val, curr_augmentation_val/self.BW, self.SFREQ)
                        # print("\t\t\tSADataset.apply_augmentations: b == ", b)
                        # print("\t\t\tSADataset.apply_augmentations: a == ", a)
                        x[:,j] = lfilter(b, a, x[:,j])
                        # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                    elif curr_augmentation == 'time_shift':
                        # print("\n\t\t\tSADataset.apply_augmentations: NOW APPLYING time_shift AUGMENTATION")
                        if curr_augmentation_val != 0:
                            # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                            new_signal = np.zeros(x[:,j].shape)
                            # print("\t\t\tSADataset.apply_augmentations: new_signal shape == ", new_signal.shape)
                            # print("\t\t\tSADataset.apply_augmentations: curr_augmentation_val == ", curr_augmentation_val)
                            if curr_augmentation_val < 0:
                                # print("\t\t\tSADataset.apply_augmentations: new_signal[:curr_augmentation_val] shape == ", new_signal[:curr_augmentation_val].shape)
                                # print("\t\t\tSADataset.apply_augmentations: x[np.abs(curr_augmentation_val):,j] shape == ", x[np.abs(curr_augmentation_val):,j].shape)
                                new_signal[:curr_augmentation_val] = x[np.abs(curr_augmentation_val):,j]
                                # print("\t\t\tSADataset.apply_augmentations: new_signal[curr_augmentation_val:] shape == ", new_signal[curr_augmentation_val:].shape)
                                # print("\t\t\tSADataset.apply_augmentations: x[:np.abs(curr_augmentation_val),j] shape == ", x[:np.abs(curr_augmentation_val),j].shape)
                                new_signal[curr_augmentation_val:] = x[:np.abs(curr_augmentation_val),j]
                            else:
                                # print("\t\t\tSADataset.apply_augmentations: new_signal[:curr_augmentation_val] shape == ", new_signal[:curr_augmentation_val].shape)
                                # print("\t\t\tSADataset.apply_augmentations: x[-curr_augmentation_val:,j] shape == ", x[-curr_augmentation_val:,j].shape)
                                new_signal[:curr_augmentation_val] = x[-curr_augmentation_val:,j]
                                # print("\t\t\tSADataset.apply_augmentations: new_signal[curr_augmentation_val:] shape == ", new_signal[curr_augmentation_val:].shape)
                                # print("\t\t\tSADataset.apply_augmentations: x[:-curr_augmentation_val,j] shape == ", x[:-curr_augmentation_val,j].shape)
                                new_signal[curr_augmentation_val:] = x[:-curr_augmentation_val,j]
                            # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                            # print("\t\t\tSADataset.apply_augmentations: new_signal shape == ", new_signal.shape)
                            x[:,j] = new_signal
                            # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                        # else:
                        #     print("\t\t\tSADataset.apply_augmentations: curr_augmentation_val == 0 -> SKIPPING")
                    elif curr_augmentation == 'zero-masking': 
                        # print("\n\t\t\tSADataset.apply_augmentations: NOW APPLYING zero-masking AUGMENTATION")
                        # print("\t\t\tSADataset.apply_augmentations: x[curr_augmentation_val[1]:curr_augmentation_val[1]+curr_augmentation_val[0], j] shape == ", x[curr_augmentation_val[1]:curr_augmentation_val[1]+curr_augmentation_val[0], j].shape)
                        x[curr_augmentation_val[1]:curr_augmentation_val[1]+curr_augmentation_val[0], j] = 0.
                    else:
                        raise NotImplementedError("curr_augmentation == "+str(curr_augmentation)+" not recognized for application")
        
        # print("SADataset.apply_augmentations: x shape == ", x.shape)
        return x
    

# UPSTREAM DATA LOADER BLOCK #################################################################################
def load_SSL_Dataset(task_id, cached_datasets_list_dir=None, total_points_val=2000, tpos_val=30, tneg_val=120, 
                     window_size=3, sfreq=1000, Nc=10, Np=16, Nb=4, max_Nb_iters=1000, total_points_factor=0.05, 
                     bw=5, randomized_augmentation=False, num_channels=11, temporal_len=3000, # items relevant to SQ/SA data loading
                     NUM_AUGMENTATIONS=2, perturb_orig_signal=True, # items unique to SA data loading
                     windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                     windowed_start_time_name="_Windowed_StartTime.npy", data_folder_name="Mouse_Training_Data", 
                     data_root_name="Windowed_Data", file_names_list="training_names.txt", 
                     train_portion=0.7, val_portion=0.2, test_portion=0.1):
    assert task_id in ['RP', 'TS', 'CPC', 'PS', 'SQ', 'SA']
    root = os.path.join(data_folder_name, data_root_name, "")

    datasets_list = []

    print("load_SSL_Dataset: loading data")
    if cached_datasets_list_dir is not None:
        cached_datasets_file_names = os.listdir(cached_datasets_list_dir)
        for name in cached_datasets_file_names:
            curr_path = os.path.join(cached_datasets_list_dir, name, "")
            if task_id == 'SA':
                datasets_list.append(SQDataset(curr_path))
            else:
                raise ValueError("load_SSL_Dataset: task_id == "+str(task_id)+" not recognized")
    else:
        f = open(os.path.join(file_names_list), 'r')
        lines = f.readlines()
        print("load_SSL_Dataset: ASSUMING EACH LINE IN ", str(file_names_list), " REPRESENTS A DIFFERENT, INDIVIDUAL TEST SUBJECT")
        for i, line in enumerate(lines):
            record_name = line.strip()
            # print("load_SSL_Dataset: processing ", record_name)
            data_file = root+record_name+os.sep+record_name
            if task_id == 'SA':
                # print("task_id == SA")
                datasets_list.append(
                    SADataset(None, 
                              path=data_file, 
                              total_points=total_points_val, 
                              window_size=window_size, 
                              sfreq=sfreq, 
                              bw=bw, 
                              randomized_augmentation=randomized_augmentation, 
                              num_channels=num_channels, 
                              temporal_len=temporal_len, 
                              NUM_AUGMENTATIONS=NUM_AUGMENTATIONS, 
                              perturb_orig_signal=perturb_orig_signal, 
                              individual_id=i, 
                              num_individuals_in_label=len(lines), 
                              windowed_data_name=windowed_data_name, 
                              windowed_start_time_name=windowed_start_time_name
                    )
                )
            else:
                raise ValueError("load_SSL_Dataset: task_id == "+str(task_id)+" not recognized")
        f.close()
    
    combined_dataset = torch.utils.data.ConcatDataset(datasets_list)

    data_len = len(combined_dataset)
    train_len = int(data_len*train_portion)
    val_len = int(data_len*val_portion)
    test_len = int(data_len - (train_len+val_len))

    # see https://pytorch.org/docs/stable/data.html
    train_set, val_set, test_set = torch.utils.data.random_split(combined_dataset, 
                                                                 [train_len, val_len, test_len], 
                                                                 generator=torch.Generator().manual_seed(0)
    )
    return train_set, val_set, test_set

def load_SSL_Dataset_Based_On_Subjects(task_id, cached_datasets_list_dir=None, total_points_val=2000, tpos_val=30, tneg_val=120, 
                                        window_size=3, sfreq=1000, Nc=10, Np=16, Nb=4, max_Nb_iters=1000, total_points_factor=0.05, 
                                        bw=5, randomized_augmentation=False, num_channels=11, temporal_len=3000, # items relevant to SQ/SA data loading
                                        NUM_AUGMENTATIONS=2, perturb_orig_signal=True, # items unique to SA data loading
                                        windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                                        windowed_start_time_name="_Windowed_StartTime.npy", data_folder_name="Mouse_Training_Data", 
                                        data_root_name="Windowed_Data", file_names_list="training_names.txt", 
                                        train_portion=0.7, val_portion=0.2, test_portion=0.1, random_seed=0):
    assert task_id in ['RP', 'TS', 'CPC', 'PS', 'SQ', 'SA']
    root = os.path.join(data_folder_name, data_root_name, "")

    datasets_list = []

    print("load_SSL_Dataset_Based_On_Subjects: loading data")
    if cached_datasets_list_dir is not None:
        cached_datasets_file_names = os.listdir(cached_datasets_list_dir)
        for name in cached_datasets_file_names:
            curr_path = os.path.join(cached_datasets_list_dir, name, "")
            if task_id == 'SA':
                datasets_list.append(SADataset(curr_path))
            else:
                raise ValueError("load_SSL_Dataset_Based_On_Subjects: task_id == "+str(task_id)+" not recognized")
    else:
        f = open(os.path.join(file_names_list), 'r')
        lines = f.readlines()
        print("load_SSL_Dataset_Based_On_Subjects: ASSUMING EACH LINE IN ", str(file_names_list), " REPRESENTS A DIFFERENT, INDIVIDUAL TEST SUBJECT")
        for i, line in enumerate(lines):
            record_name = line.strip()
            # print("load_SSL_Dataset_Based_On_Subjects: processing ", record_name)
            data_file = root+record_name+os.sep+record_name
            if task_id == 'SA':
                # print("task_id == SA")
                datasets_list.append(
                    SADataset(None, 
                              path=data_file, 
                              total_points=total_points_val, 
                              window_size=window_size, 
                              sfreq=sfreq, 
                              bw=bw, 
                              randomized_augmentation=randomized_augmentation, 
                              num_channels=num_channels, 
                              temporal_len=temporal_len, 
                              NUM_AUGMENTATIONS=NUM_AUGMENTATIONS, 
                              perturb_orig_signal=perturb_orig_signal, 
                              individual_id=i, 
                              num_individuals_in_label=len(lines), 
                              windowed_data_name=windowed_data_name, 
                              windowed_start_time_name=windowed_start_time_name
                    )
                )
            else:
                raise ValueError("load_SSL_Dataset_Based_On_Subjects: task_id == "+str(task_id)+" not recognized")
        f.close()
    
    shuffle_inds = [i for i in range(len(datasets_list))]
    random.Random(random_seed).shuffle(shuffle_inds) # see https://stackoverflow.com/questions/19306976/python-shuffling-with-a-parameter-to-get-the-same-result
    datasets_list = datasets_list[shuffle_inds]
    
    # combined_dataset = torch.utils.data.ConcatDataset(datasets_list)

    # data_len = len(combined_dataset)
    # train_len = int(data_len*train_portion)
    # val_len = int(data_len*val_portion)
    # test_len = int(data_len - (train_len+val_len))

    # # see https://pytorch.org/docs/stable/data.html
    # train_set, val_set, test_set = torch.utils.data.random_split(combined_dataset, 
    #                                                              [train_len, val_len, test_len], 
    #                                                              generator=torch.Generator().manual_seed(0)
    # )

    num_train_subjects = max(1, int(train_portion*len(datasets_list)))
    num_val_subjects = max(1, int(val_portion*len(datasets_list)))
    num_test_subjects = max(1, int((1. - (train_portion+val_portion))*len(datasets_list)))

    train_set = torch.utils.data.ConcatDataset(datasets_list[:num_train_subjects])
    train_set = torch.utils.data.random_split(train_set, 
                                              [len(train_set)], 
                                              generator=torch.Generator().manual_seed(random_seed)
    )
    val_set = torch.utils.data.ConcatDataset(datasets_list[num_train_subjects:num_train_subjects+num_val_subjects])
    val_set = torch.utils.data.random_split(val_set, 
                                            [len(val_set)], 
                                            generator=torch.Generator().manual_seed(random_seed)
    )
    test_set = torch.utils.data.ConcatDataset(datasets_list[num_train_subjects+num_val_subjects:])
    test_set = torch.utils.data.random_split(test_set, 
                                             [len(test_set)], 
                                             generator=torch.Generator().manual_seed(random_seed)
    )

    train_val_test_split_subjects = {"train_set_subjects": shuffle_inds[:num_train_subjects], 
                                     "val_set_subjects": shuffle_inds[num_train_subjects:num_train_subjects+num_val_subjects], 
                                     "test_set_subjects": shuffle_inds[num_train_subjects+num_val_subjects:], 
    }

    return train_set, val_set, test_set, train_val_test_split_subjects
# UPSTREAM DATA LOADER BLOCK #################################################################################

