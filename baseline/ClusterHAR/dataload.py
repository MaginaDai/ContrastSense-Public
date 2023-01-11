import torch
import sys
import numpy as np
import os
import random
from scipy.interpolate import interp1d
from os.path import dirname
sys.path.append(dirname(dirname(sys.path[0])))

from baseline.CLHAR.CL_dataload import CL_Rotate
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from data_aug.contrastive_learning_dataset import ACT_Translated_labels, HHAR_movement, fetch_dataset_root, users, devices
from data_aug.preprocessing import UsersPosition
from data_aug.view_generator import ContrastiveLearningViewGenerator, SingleViewGenerator
from data_aug import imu_transforms
from exceptions.exceptions import InvalidDatasetSelection



class Resampling(object):
    """ https://arxiv.org/abs/2109.02054 
        adapted from https://github.com/diheal/resampling
    """
    def __call__(self, sensor):
        M, N = random.choice([[1, 0], [2, 1], [3, 2]])

        time_steps = sensor.shape[0]
        raw_set = np.arange(sensor.shape[0])
        interp_steps = np.arange(0, raw_set[-1] + 1e-1, 1 / (M + 1))
        
        x_interp = interp1d(raw_set, sensor, axis=0)
        x_up = x_interp(interp_steps)

        length_inserted = x_up.shape[0]
        start = random.randint(0, length_inserted - time_steps * (N + 1))
        index_selected = np.arange(start, start + time_steps * (N + 1), N + 1)
        return x_up[index_selected, :]


class ClusterCLDataset:
    def __init__(self, transfer, version, datasets_name=None):
        self.transfer = transfer
        self.datasets_name = datasets_name
        self.version = version

    def get_simclr_pipeline_transform(self):
        """Return a set of data augmentation transformations as described in my presentation."""
        imu_toTensor = imu_transforms.ToTensor()
        imu_rotate = CL_Rotate()
        # imu_resample = Resampling()
        data_transforms = transforms.Compose([imu_rotate,
                                              imu_toTensor])
        return data_transforms

    def get_dataset(self, split, n_views=2, percent=20, shot=None):
        if self.transfer is False:
            # here it is for generating positive samples and negative samples
            return ClusterHARDataset4Training(self.datasets_name, self.version, n_views, 
                                    transform=SingleViewGenerator(imu_transforms.ToTensor(), self.get_simclr_pipeline_transform()),
                                    split=split, transfer=self.transfer, percent=percent, shot=shot)
        elif split == 'train' or split == 'tune':
                # here it is for data augmentation, make it more challenging.
            return ClusterHARDataset4Training(self.datasets_name, self.version, n_views,
                                    self.get_simclr_pipeline_transform(),
                                    split=split, transfer=self.transfer, percent=percent, shot=shot)
        else:  # val or test
            return ClusterHARDataset4Training(self.datasets_name, self.version, n_views,
                                    transform=transforms.Compose([
                                        imu_transforms.ToTensor()
                                        ]),
                                    split=split, transfer=self.transfer, percent=percent, shot=shot)


class ClusterHARDataset4Training(Dataset):

    def __init__(self, datasets_name, version, n_views=2, transform=None, split='train', transfer=True, percent=20, shot=None):
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
            if shot:
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
        self.n_views = n_views

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
        acc, gyro, label = sample['acc'], sample['gyro'], sample['add_infor']
        sensor = np.concatenate((acc, gyro), axis=1)
        return sensor, torch.from_numpy(label).long()