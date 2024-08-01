from contrastive_training import BatchedRandomisedDataset, ConcatenatedDataset, ZippedDataset, ceiling_division, get_group_held_out_users, shuffle_array
import numpy as np

from data_aug.contrastive_learning_dataset import fetch_dataset_root
from device_selection import get_pos_neg_apriori

from torchvision.transforms import transforms
from torch.utils.data import Dataset
import load_data
import torch
import os
import sys

from os.path import dirname
sys.path.append(dirname(dirname(sys.path[0])))


class ColloSSL_Dataset(object):
    
    def __init__(self, args, all_devices, transfer, version, datasets_name=None):
        self.transfer = transfer
        self.datasets_name = datasets_name
        self.version = version
        self.all_devices = all_devices
        self.args = args

        self.dataset_full = load_data.Data(args.dataset_path, args.dataset_name, load_path=None, held_out=args.held_out)
        
        input_shape = self.dataset_full.input_shape

        output_shape = len(self.dataset_full.info['session_list'])
        print("input shape", input_shape)
        print("output shape", output_shape)


    def get_dataset(self, split, percent=20, shot=None):
        version = self.version
        transfer=self.transfer
        percent=percent
        shot=shot
        batch_size=self.args.batch_size

        datasets_name = self.datasets_name
        root_dir = fetch_dataset_root(datasets_name)

        train_dir = '../../' + root_dir + '_col' + version + '/train_set.npz'  # we store the users splits in those correpsonding results. 
        val_dir = '../../' + root_dir + '_col' + version + '/val_set.npz'
        test_dir = '../../' + root_dir + '_col' + version + '/test_set.npz'

        self.datasets_name = datasets_name
        self.transfer = transfer
        self.split = split
        self.batch_size = batch_size
        if self.split == 'train':
            data = np.load(train_dir)
            self.windows_frame = data['train_set']
        elif self.split == 'val':
            data = np.load(val_dir)
            self.windows_frame = data['val_set']
        elif self.split == 'tune':
            if shot >= 0:
                tune_dir = '../../' + root_dir + '_' + version + '/tune_set_' + str(int(shot)) + '.npz'
            else:
                if percent <= 0.99:
                    tune_dir = '../../' + root_dir + '_' + version + '/tune_set_' + str(percent).replace('.', '_') + '.npz'
                else:
                    tune_dir = '../../' + root_dir + '_' + version + '/tune_set_' + str(int(percent)) + '.npz'
            data = np.load(tune_dir)
            self.windows_frame = data['tune_set']
        else:
            data = np.load(test_dir)
            self.windows_frame = data['test_set']
        self.root_dir = root_dir
        

        ######### 
        included_user = None


        anchor_index = 0
        positive_indices = np.arange(len(self.args.positive_devices)) + 1
        negative_indices = np.arange(len(self.args.negative_devices)) + 1 + len(self.args.positive_devices)
        

        training_set_device = [(np.concatenate([self.dataset_full.device_user_ds[d][u][0] for u in self.dataset_full.device_user_ds[d] if u in included_user], axis=0)) for d in self.all_devices]

        distances = None
                        
        user_device_dataset = []
        for u in self.dataset_full.info['user_list']:
            dataset_per_user = []
            if u in included_user:
            # if args.held_out is None or u != dataset_full.info['user_list'][args.held_out]:
                for d in self.all_devices:
                    X = self.dataset_full.device_user_ds[d][u][0]
                    len_x = X.shape[0]
                    X_shuffled = shuffle_array(X, seed=42, inplace=False) 
                    dataset_per_user.append(X_shuffled[: int(len_x)])
                user_device_dataset.append(dataset_per_user)

        tf_train_contrast_list = []
        for user_dataset in user_device_dataset:
            device_dataset_shuffled = []
            for device_index, device_dataset in enumerate(user_dataset):

                if device_index == anchor_index:
                    seed = 42
                elif device_index in positive_indices:
                    seed = 42
                else: 
                    seed = 43 + device_index
                        
                #For Dynamic Device Selection, we want the datasets to be synced here. They will be shuffled later
                if self.args.dynamic_device_selection==1 and self.args.multi_sampling_mode != 'unsync_all':  
                    seed = 42

                shuffled = BatchedRandomisedDataset_torch(device_dataset, batch_size=self.batch_size, seed=seed)
                device_dataset_shuffled.append(shuffled)

            tf_train_contrast_list.append(ZippedDataset_torch(device_dataset_shuffled, stack_batches=True))

        self.dataset = ConcatenatedDataset_torch(tf_train_contrast_list)
        return self.dataset

    def __len__(self):
        return len(self.dataset)


class BatchedRandomisedDataset_torch(Dataset):
    def __init__(self, data, batch_size, seed=42, randomised=True, axis=0, name=""):
        self.name = name
        self.data = data
        self.batch_size = batch_size
        self.axis = axis
        self.seed = seed
        self.data_len = data.shape[self.axis]
        self.num_batches = ceiling_division(self.data_len, batch_size)
        self.randomised = randomised
        self.rng = np.random.default_rng(seed=seed)

        
    def reset_dataset(self):
        if self.randomised:
            index_list = np.arange(self.data_len, dtype=int)
            self.rng.shuffle(index_list)
            self.shuffled_dataset = self.data[index_list]
            self.output_dataset = self.shuffled_dataset

        if not self.axis == 0:
            self.output_dataset = np.moveaxis(self.output_dataset, self.axis, 0)
        self.i = 0

    def __len__(self):
        return self.num_batches

    def __getitem__(self):
        self.reset_dataset()

        for i in range(self.num_batches):
            return np.moveaxis(self.output_dataset[i * self.batch_size : (i + 1) * self.batch_size], 0, self.axis)


class SequenceEagerZippedDataset_torch(Dataset):
    def __init__(self, batched_randomised_datasets, stack_batches=True, stack_axis=0):
        self.datasets = batched_randomised_datasets
        self.stack_batches = stack_batches
        self.stack_axis = stack_axis
        self.reset_dataset()

    def reset_dataset(self):
        if self.stack_batches:
            self.data = [
                np.stack(zipped_batch, axis=self.stack_axis) for zipped_batch in zip(*tuple(self.datasets))
            ]
        else:
            self.data = [
                zipped_batch for zipped_batch in zip(*tuple(self.datasets))
            ]

    def __len__(self):
        return min([len(dataset) for dataset in self.datasets])
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def on_epoch_end(self):
        self.reset_dataset()


class ZippedDataset_torch(Dataset):
    def __init__(self, batched_randomised_datasets, stack_batches=True, stack_axis=0):
        self.datasets = batched_randomised_datasets
        self.stack_batches = stack_batches
        self.stack_axis = stack_axis

    def __len__(self):
        return min([len(dataset) for dataset in self.datasets])

    def __getitem__(self):
        if self.stack_batches:
            for zipped_batch in zip(*tuple(self.datasets)):
                return np.stack(zipped_batch, axis=self.stack_axis)
        else:
            for zipped_batch in zip(*tuple(self.datasets)):
                return zipped_batch

class ConcatenatedDataset_torch(Dataset):
    def __init__(self, batched_randomised_datasets):
        self.datasets = batched_randomised_datasets
        self.num_datasets = len(self.datasets)

    
    def __len__(self):
        return min([len(dataset) for dataset in self.datasets])
    
    def __getitem__(self):
        for dataset in self.datasets:
            for batch in dataset:
                return batch