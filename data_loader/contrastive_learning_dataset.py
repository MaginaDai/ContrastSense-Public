import torch
import os
import pdb

from data_loader import imu_transforms, emg_transforms
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision import transforms
from data_loader.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
import numpy as np



def fetch_dataset_root(dataset_name):
    root = {'HHAR': './datasets/HHAR',
            'MotionSense': './datasets/MotionSense',
            'Shoaib': './datasets/Shoaib',
            'HASC': './datasets/HASC',
            'Myo': './datasets/Myo',
            'NinaPro': './datasets/NinaPro',
            'Merged_dataset': './datasets/Merged_dataset',
            }
    try:
        root_dir = root[dataset_name]
    except KeyError:
        raise InvalidDatasetSelection()
    else:
        return root_dir
    

class ContrastiveLearningDataset:
    def __init__(self, transfer, version, datasets_name=None, cross_dataset=False, modal='imu'):
        """
        Args:
            transfer : pretrain or transfer 
            version: dataset split version 
            datasets_name: dataset name
            cross_dataset: whether do cross dataset experiment or not
            modal: the modality (IMU or EMG)
        """

        self.transfer = transfer
        self.datasets_name = datasets_name
        self.version = version
        self.cross_dataset = cross_dataset
        self.modal = modal


    def get_imu_pipeline_transform(self):
        """Return a set of data augmentation transformations."""

        imu_noise = imu_transforms.IMUNoise(var=0.05, p=0.8)
        imu_scale = imu_transforms.IMUScale(scale=[0.9, 1.1], p=0.8)
        imu_rotate = imu_transforms.IMURotate(p=0.8)
        imu_negate = imu_transforms.IMUNegated(p=0.4)
        imu_flip = imu_transforms.IMUHorizontalFlip(p=0.2)
        imu_warp = imu_transforms.IMUTimeWarp(p=0.4)

        imu_toTensor = imu_transforms.ToTensor()

        data_transforms = transforms.Compose([imu_scale,
                                              imu_rotate,
                                              imu_negate,
                                              imu_flip,
                                              imu_warp,
                                              imu_noise,
                                              imu_toTensor])
        return data_transforms
    
    def get_ft_pipeline_transform(self):
        imu_noise = imu_transforms.IMUNoise(var=0.05, p=0.8)
        imu_scale = imu_transforms.IMUScale(scale=[0.9, 1.1], p=0.8)
        imu_rotate = imu_transforms.IMURotate(p=0.8) 
        imu_negate = imu_transforms.IMUNegated(p=0.4)
        imu_flip = imu_transforms.IMUHorizontalFlip(p=0.2)
        imu_warp = imu_transforms.IMUTimeWarp(p=0.4)
        imu_toTensor = imu_transforms.ToTensor()

        data_transforms = transforms.Compose([imu_scale,
                                              imu_rotate,
                                              imu_negate,
                                              imu_flip,
                                              imu_warp,
                                              imu_noise,
                                              imu_toTensor])
        return data_transforms
    
    def get_emg_pipeline_transform(self):  # need to rewrite for emg
        emg_noise = emg_transforms.EMGNoise(var=0.05, p=0.8)
        emg_scale = emg_transforms.EMGScale(scale=[0.9, 1.1], p=0.8)
        emg_flip = emg_transforms.EMGHorizontalFlip(p=0.2)
        emg_negate = emg_transforms.EMGNegated(p=0.4)
        emg_warp = emg_transforms.EMGTimeWarp(p=0.4)
        emg_toTensor = emg_transforms.EMGToTensor()

        data_transforms = transforms.Compose([emg_scale,
                                              emg_negate,
                                              emg_flip,
                                              emg_warp,
                                              emg_noise,
                                              emg_toTensor])
        return data_transforms

    def get_dataset(self, split, n_views=2, percent=20, shot=None):
        if not self.transfer or split == 'train' or split == 'tune':
            if self.modal == 'imu':
                transformation = self.get_imu_pipeline_transform()
            elif self.modal == 'emg':
                transformation = self.get_emg_pipeline_transform()
            else:
                NotADirectoryError
        else:
            if self.modal == 'imu':
                transformation = transforms.Compose([imu_transforms.ToTensor()])
            elif self.modal == 'emg':
                transformation = transforms.Compose([emg_transforms.EMGToTensor()])
            else:
                NotADirectoryError
        
        if self.transfer is False:
            # here it is for generating positive samples and negative samples
            return Dataset4Training(self.datasets_name, self.version, n_views, 
                                    transform=ContrastiveLearningViewGenerator(transformation, n_views),
                                    split=split, transfer=self.transfer, percent=percent, 
                                    shot=shot, cross_dataset=self.cross_dataset, modal=self.modal)
        else: # tune/valition/test
            return Dataset4Training(self.datasets_name, self.version, n_views, transformation,
                                    split=split, transfer=self.transfer, percent=percent,
                                    shot=shot, cross_dataset=self.cross_dataset, modal=self.modal)


class Dataset4Training(Dataset):

    def __init__(self, datasets_name, version, n_views=2, transform=None, split='train', transfer=True, percent=20, shot=None, cross_dataset=False, modal='imu'):
        """
        Args:
            datasets_name (string): dataset name.
            transform (callable, optional): Optional transform to be applied on a sample.
            n_views: the defined views. defalt=2
            split: to specify train set or test set.
        """
        root_dir = fetch_dataset_root(datasets_name)
        
        train_dir = root_dir + '_' + version + '/train_set.npz'
        val_dir = root_dir + '_' + version + '/val_set.npz'
        test_dir = root_dir + '_' + version + '/test_set.npz'

        self.datasets_name = datasets_name
        self.transfer = transfer
        self.split = split
        self.modal = modal

        if self.split == 'train':
            data = np.load(train_dir)
            self.windows_frame = data['train_set']
        elif self.split == 'val':
            data = np.load(val_dir)
            self.windows_frame = data['val_set']
        elif self.split == 'tune':
            if shot >= 0:
                tune_dir = root_dir + '_' + version + '/tune_set_' + str(int(shot)) + '.npz'
            else:
                if percent <= 0.99:
                    tune_dir = root_dir + '_' + version + '/tune_set_' + str(percent).replace('.', '_') + '.npz'
                else:
                    tune_dir = root_dir + '_' + version + '/tune_set_' + str(int(percent)) + '.npz'
            data = np.load(tune_dir)
            self.windows_frame = data['tune_set']
        else:
            data = np.load(test_dir)
            self.windows_frame = data['test_set']
        self.root_dir = root_dir
        self.transform = transform
        self.n_views = n_views

        self.cross_dataset = cross_dataset

    def __len__(self):
        return len(self.windows_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        loc = os.path.join(self.root_dir, self.windows_frame[idx])

        sample = np.load(loc, allow_pickle=True)

        sensor, label = self.sperate_label_data(sample)
        if self.transform:
            sensor = self.transform(sensor)
        return sensor, label


    def sperate_label_data(self, sample):
        if self.modal == 'imu':
            acc, gyro, label = sample['acc'], sample['gyro'], sample['add_infor']
            sensor = np.concatenate((acc, gyro), axis=1)

        elif self.modal == 'emg':
            sensor, label = sample['emg'], sample['add_infor']
        elif self.modal == 'eeg':
            sensor, label = sample['eeg'], sample['add_infor']

        return sensor, torch.from_numpy(label).long()
