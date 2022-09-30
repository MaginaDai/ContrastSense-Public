import torch
import sys
import numpy as np
import os
from os.path import dirname
sys.path.append(dirname(dirname(sys.path[0])))

from torchvision.transforms import transforms
from torch.utils.data import Dataset
from data_aug.contrastive_learning_dataset import ACT_Translated_labels, HHAR_movement, fetch_dataset_root, users, devices
from data_aug.preprocessing import UsersPosition
from data_aug.view_generator import ContrastiveLearningViewGenerator, SingleViewGenerator
from data_aug import imu_transforms
from exceptions.exceptions import InvalidDatasetSelection




class ClusterCLDataset:
    def __init__(self, transfer, M, N, version, datasets_name=None):
        self.transfer = transfer
        self.datasets_name = datasets_name
        self.M = M
        self.N = N
        self.version = version

    def get_simclr_pipeline_transform(self):
        """Return a set of data augmentation transformations as described in my presentation."""
        imu_toTensor = imu_transforms.ToTensor()
        imu_resample = imu_transforms.IMU_Resampling(M=self.M, N=self.N)
        data_transforms = transforms.Compose([imu_resample,
                                              imu_toTensor])
        return data_transforms

    def get_dataset(self, split, n_views=2, percent=20):
        if self.transfer is False:
            # here it is for generating positive samples and negative samples
            return ClusterHARDataset4Training(self.datasets_name, self.version, n_views, 
                                    transform=SingleViewGenerator(imu_transforms.ToTensor(), self.get_simclr_pipeline_transform()),
                                    split=split, transfer=self.transfer, percent=percent)
        elif split == 'train' or split == 'tune':
                # here it is for data augmentation, make it more challenging.
            return ClusterHARDataset4Training(self.datasets_name, self.version, n_views,
                                    self.get_simclr_pipeline_transform(),
                                    split=split, transfer=self.transfer, percent=percent)
        else:  # val or test
            return ClusterHARDataset4Training(self.datasets_name, self.version, n_views,
                                    transform=transforms.Compose([
                                        imu_transforms.ToTensor()
                                        ]),
                                    split=split, transfer=self.transfer, percent=percent)


class ClusterHARDataset4Training(Dataset):

    def __init__(self, datasets_name, version, n_views=2, transform=None, split='train', transfer=True, percent=20):
        """
        Args:
            datasets_name (string): dataset name.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            n_views: the defined views. defalt=2
            split: to specify train set or test set.
        """
        root_dir = fetch_dataset_root(datasets_name, version)

        train_dir = '../../' + root_dir + '/train_set.npz'
        val_dir = '../../' + root_dir + '/val_set.npz'
        test_dir = '../../' + root_dir + '/test_set.npz'

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
            if percent <= 0.99:
                tune_dir = '../../' + root_dir + '/tune_set_' + str(percent).replace('.', '_') + '.npz'
            else:
                tune_dir = '../../' + root_dir + '/tune_set_' + str(int(percent)) + '.npz'
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

        # if not self.transfer:
        #     sensor = [sample[i]['sensor'] for i in range(self.n_views)]
        #     return sensor, label
        # else:
        #     return sensor, label
        return sensor, label
    
    def sperate_label_data(self, sample):
        acc, gyro, add_infor = sample['acc'], sample['gyro'], sample['add_infor']
        if 'HHAR' in self.datasets_name:
            label = np.array([HHAR_movement.index(add_infor[0, -1]), users.index(add_infor[0, -3]), devices.index(add_infor[0, -2])])
        elif self.datasets_name == 'MotionSense':
            label = np.array([ACT_Translated_labels.index(add_infor[0, -1]), int(add_infor[0, UsersPosition[self.datasets_name]])])
        elif self.datasets_name == 'UCI':
            label = np.array([int(add_infor[0, -2]), int(add_infor[0, UsersPosition[self.datasets_name]])])
        elif self.datasets_name == 'Shoaib':
            label = np.array([int(add_infor[0, -2]), int(add_infor[0, UsersPosition[self.datasets_name]])])
        else:
            raise InvalidDatasetSelection()
        sensor = np.concatenate((acc, gyro), axis=1)
        return sensor, torch.from_numpy(label).long()