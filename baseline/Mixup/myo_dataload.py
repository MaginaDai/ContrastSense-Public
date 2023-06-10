from torchvision.transforms import transforms
from torch.utils.data import Dataset

from data_aug.contrastive_learning_dataset import fetch_dataset_root

import numpy as np
import torch
import os

class Myo_Dataset:
    def __init__(self, transfer, version, datasets_name=None):
        self.transfer = transfer
        self.datasets_name = datasets_name
        self.version = version

    def get_dataset(self, split, percent=20, shot=None):
        self.dataset = MyoDataset4Training(self.datasets_name, self.version,
                                           transform=transforms.Compose([ToTensor()]),
                                           split=split, transfer=self.transfer, percent=percent, shot=shot)
        return self.dataset

    def __len__(self):
        return len(self.dataset)


class MyoDataset4Training(Dataset):

    def __init__(self, datasets_name, version, transform=None, split='train', transfer=True, percent=20, shot=None):
        """
        Args:
            datasets_name (string): dataset name.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            n_views: the defined views. defalt=2
            split: to specify train set or test set.
        """
        root_dir = fetch_dataset_root(datasets_name)

        train_dir = '../../' + root_dir + '_' + version + '/train_set.npz'
        val_dir = '../../' + root_dir + '_' + version + '/val_set.npz'
        test_dir = '../../' + root_dir + '_' + version + '/test_set.npz'

        self.datasets_name = datasets_name
        self.transfer = transfer
        self.split = split
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
        self.transform = transform

    def __len__(self):
        return len(self.windows_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        loc = os.path.join('../../' + self.root_dir,
                           self.windows_frame[idx])

        sample = np.load(loc, allow_pickle=True)

        sensor, label = self.sperate_label_data(sample)

        if self.transform:
            sensor = self.transform(sensor)

        return sensor, label
    
    def sperate_label_data(self, sample):
        sensor, label = sample['emg'], sample['add_infor']
        return sensor.astype('float32'), torch.from_numpy(label).long()
    

class ToTensor(object):
    def __call__(self, sensor):
        return torch.from_numpy(sensor)