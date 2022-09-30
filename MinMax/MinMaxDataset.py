import torch
import os

from MinMax import IMU_Trans_MinMax
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from figure_plot.figure_plot import figure_cmp_transformation


class MinMaxDataset:
    def __init__(self, root_folder, transfer, datasets_name='UciHarDataset'):
        self.root_folder = root_folder
        self.transfer = transfer
        self.datasets_name = datasets_name

    def basic_transform(self):
        return transforms.Compose([
            IMU_Trans_MinMax.IMUDiscard(self.datasets_name)
        ])

    def get_dataset(self, name, split, percent=20, n_views=2):
        valid_datasets = {'HHAR': lambda: HHAR_Dataset(self.root_folder, n_views, transform=self.basic_transform(),
                                                       split=split, transfer=self.transfer, percent=percent),
                          'MotionSense': lambda: MotionSense(self.root_folder, n_views, transform=self.basic_transform(),
                                                             split=split, transfer=self.transfer, percent=percent),
                          'UCI': lambda: UCI(self.root_folder, n_views, transform=self.basic_transform(),
                                             split=split, transfer=self.transfer, percent=percent),
                          'Shoaib': lambda: Shoaib(self.root_folder, n_views, transform=self.basic_transform(),
                                                   split=split, transfer=self.transfer, percent=percent)
                          }
        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()


class HHAR_Dataset(Dataset):
    """ HHAR dataset, Huatao preprocessing way """

    def __init__(self, root_dir, n_views=2, transform=None, split='train', transfer=True, percent=20):
        """
        Args:
            root_dir (string): Path to all the npz file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            n_views: the defined views. defalt=2
            split: to specify train set or test set.
        """

        train_dir = r'./datasets/HHAR/train_set.npz'
        val_dir = r'./datasets/HHAR/val_set.npz'
        test_dir = r'./datasets/HHAR/test_set.npz'

        self.transfer = transfer
        self.split = split
        if self.split == 'train':
            data = np.load(train_dir)
            self.windows_frame = data['train_set']
        elif self.split == 'val':
            data = np.load(val_dir)
            self.windows_frame = data['val_set']
        elif self.split == 'tune':
            if percent <= 0.99:
                tune_dir = r'./datasets/HHAR/tune_set_' + str(percent).replace('.', '_') + '.npz'
            else:
                tune_dir = r'./datasets/HHAR/tune_set_' + str(int(percent)) + '.npz'
            data = np.load(tune_dir)
            self.windows_frame = data['tune_set']
        else:
            data = np.load(test_dir)
            self.windows_frame = data['test_set']

        self.root_dir = root_dir
        self.transform = transform
        self.n_views = n_views

    def __len__(self):
        return len(self.windows_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        loc = os.path.join(self.root_dir,
                           self.windows_frame[idx])

        sample = np.load(loc, allow_pickle=True)

        if self.transform:
            sample = self.transform(sample)

        sensor = sample['sensor']
        return sensor, sample['label']


class MotionSense(Dataset):
    """ MotionSense dataset, Huatao preprocessing way """

    def __init__(self, root_dir, n_views=2, transform=None, split='train', transfer=True, percent=20):
        """
        Args:
            root_dir (string): Path to all the npz file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            n_views: the defined views. defalt=2
            split: to specify train set or test set.
        """

        train_dir = r'./datasets/MotionSense/train_set.npz'
        val_dir = r'./datasets/MotionSense/val_set.npz'
        test_dir = r'./datasets/MotionSense/test_set.npz'

        self.transfer = transfer
        self.split = split
        if self.split == 'train':
            data = np.load(train_dir)
            self.windows_frame = data['train_set']
        elif self.split == 'val':
            data = np.load(val_dir)
            self.windows_frame = data['val_set']
        elif self.split == 'tune':
            if percent <= 0.99:
                tune_dir = r'./datasets/MotionSense/tune_set_' + str(percent).replace('.', '_') + '.npz'
            else:
                tune_dir = r'./datasets/MotionSense/tune_set_' + str(int(percent)) + '.npz'
            data = np.load(tune_dir)
            self.windows_frame = data['tune_set']
        else:
            data = np.load(test_dir)
            self.windows_frame = data['test_set']

        self.root_dir = root_dir
        self.transform = transform
        self.n_views = n_views

    def __len__(self):
        return len(self.windows_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        loc = os.path.join(self.root_dir,
                           self.windows_frame[idx])

        sample = np.load(loc, allow_pickle=True)

        if self.transform:
            sample = self.transform(sample)

        sensor = sample['sensor']
        return sensor, sample['label']


class UCI(Dataset):
    """ UCI dataset, Huatao preprocessing way """

    def __init__(self, root_dir, n_views=2, transform=None, split='train', transfer=True, percent=20):
        """
        Args:
            root_dir (string): Path to all the npz file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            n_views: the defined views. defalt=2
            split: to specify train set or test set.
        """

        train_dir = r'./datasets/UCI/train_set.npz'
        val_dir = r'./datasets/UCI/val_set.npz'
        test_dir = r'./datasets/UCI/test_set.npz'

        self.transfer = transfer
        self.split = split
        if self.split == 'train':
            data = np.load(train_dir)
            self.windows_frame = data['train_set']
        elif self.split == 'val':
            data = np.load(val_dir)
            self.windows_frame = data['val_set']
        elif self.split == 'tune':
            if percent <= 0.99:
                tune_dir = r'./datasets/UCI/tune_set_' + str(percent).replace('.', '_') + '.npz'
            else:
                tune_dir = r'./datasets/UCI/tune_set_' + str(int(percent)) + '.npz'
            data = np.load(tune_dir)
            self.windows_frame = data['tune_set']
        else:
            data = np.load(test_dir)
            self.windows_frame = data['test_set']

        self.root_dir = root_dir
        self.transform = transform
        self.n_views = n_views

    def __len__(self):
        return len(self.windows_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        loc = os.path.join(self.root_dir,
                           self.windows_frame[idx])

        sample = np.load(loc, allow_pickle=True)

        if self.transform:
            sample = self.transform(sample)

        if not self.transfer:
            sensor = [sample[i]['sensor'] for i in range(self.n_views)]
            return sensor, sample[0]['label']
        else:
            sensor = sample['sensor']
            return sensor, sample['label']
        # return sample


class Shoaib(Dataset):
    """ Shoaib dataset, Huatao preprocessing way """

    def __init__(self, root_dir, n_views=2, transform=None, split='train', transfer=True, percent=20):
        """
        Args:
            root_dir (string): Path to all the npz file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            n_views: the defined views. defalt=2
            split: to specify train set or test set.
        """

        train_dir = r'./datasets/Shoaib/train_set.npz'
        val_dir = r'./datasets/Shoaib/val_set.npz'
        test_dir = r'./datasets/Shoaib/test_set.npz'

        self.transfer = transfer
        self.split = split
        if self.split == 'train':
            data = np.load(train_dir)
            self.windows_frame = data['train_set']
        elif self.split == 'val':
            data = np.load(val_dir)
            self.windows_frame = data['val_set']
        elif self.split == 'tune':
            if percent <= 0.99:
                tune_dir = r'./datasets/Shoaib/tune_set_' + str(percent).replace('.', '_') + '.npz'
            else:
                tune_dir = r'./datasets/Shoaib/tune_set_' + str(int(percent)) + '.npz'
            data = np.load(tune_dir)
            self.windows_frame = data['tune_set']
        else:
            data = np.load(test_dir)
            self.windows_frame = data['test_set']

        self.root_dir = root_dir
        self.transform = transform
        self.n_views = n_views

    def __len__(self):
        return len(self.windows_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        loc = os.path.join(self.root_dir,
                           self.windows_frame[idx])

        sample = np.load(loc, allow_pickle=True)

        if self.transform:
            sample = self.transform(sample)

        if not self.transfer:
            sensor = [sample[i]['sensor'] for i in range(self.n_views)]
            return sensor, sample[0]['label']
        else:
            sensor = sample['sensor']
            return sensor, sample['label']
        # return sample


class UotHarDataset(Dataset):
    """ UoT HAR dataset """

    def __init__(self, root_dir, n_views=2, transform=None, split='train', transfer=True, percent=20):
        """
        Args:
            root_dir (string): Path to all the npz file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            n_views: the defined views. defalt=2
            split: to specify train set or test set.
        """

        train_dir = r'./datasets/uot_processed/train_set.npz'
        val_dir = r'./datasets/uot_processed/val_set.npz'
        test_dir = r'./datasets/uot_processed/test_set.npz'

        self.transfer = transfer
        self.split = split
        if self.split == 'train':
            data = np.load(train_dir)
            self.windows_frame = data['train_set']
        elif self.split == 'val':
            data = np.load(val_dir)
            self.windows_frame = data['val_set']
        elif self.split == 'tune':
            tune_dir = r'./datasets/uot_processed/tune_set_' + str(percent).replace('.', '_') + '.npz'
            data = np.load(tune_dir)
            self.windows_frame = data['tune_set']
        else:
            data = np.load(test_dir)
            self.windows_frame = data['test_set']

        self.root_dir = root_dir
        self.transform = transform
        self.n_views = n_views

    def __len__(self):
        return len(self.windows_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        loc = os.path.join(self.root_dir,
                           self.windows_frame[idx])

        sample = np.load(loc, allow_pickle=True)

        if self.transform:
            sample = self.transform(sample)

        if not self.transfer:
            sensor = [sample[i]['sensor'] for i in range(self.n_views)]
            return sensor, sample[0]['label']
        else:
            sensor = sample['sensor']
            return sensor, sample['label']

