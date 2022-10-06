from doctest import set_unittest_reportflags
import imp
from tkinter import N
import torch
import os
import pdb

from data_aug import imu_transforms
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision import transforms, datasets
from data_aug.preprocessing import UsersPosition
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from figure_plot.figure_plot import figure_cmp_transformation

from utils import Preprocess4Normalization,  Preprocess4Mask


HHAR_movement = ['stand', 'sit', 'walk', 'stairsup', 'stairsdown', 'bike']
ACT_Translated_labels = ['Downstairs', 'Upstairs', 'Walking', 'Running', 'Standing', 'Sitting']
users = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
devices = ['s3', 'nexus4', 's3mini', 'samsungold']


def fetch_dataset_root(dataset_name, version):
    root = {'HHAR': './datasets/HHAR',
            'MotionSense': './datasets/MotionSense',
            'UCI': './datasets/UCI',
            'Shoaib': './datasets/Shoaib',
            'ICHAR': './datasets/ICHAR',
            'HASC': './datasets/HASC'}
    try:
        root_dir = root[dataset_name] + '_' + version
    except KeyError:
        raise InvalidDatasetSelection()
    else:
        return root_dir
    

class ContrastiveLearningDataset:
    def __init__(self, transfer, version, datasets_name=None):
        self.transfer = transfer
        self.datasets_name = datasets_name
        self.version = version

    def get_simclr_pipeline_transform(self):
        """Return a set of data augmentation transformations as described in my presentation."""
        imu_discard = imu_transforms.IMUDiscard(self.datasets_name)
        imu_noise = imu_transforms.IMUNoise(var=0.05, p=0.8)
        imu_scale = imu_transforms.IMUScale(scale=[0.9, 1.1], p=0.8)
        imu_rotate = imu_transforms.IMURotate(p=0.8)
        imu_negate = imu_transforms.IMUNegated(p=0.4)
        imu_flip = imu_transforms.IMUHorizontalFlip(p=0.1)
        imu_warp = imu_transforms.IMUTimeWarp(p=0.4)

        imu_error_model = imu_transforms.IMUErrorModel(p=0.8, scale=[0.9, 1.1], error_magn=0.02, bias_magn=0.05)
        imu_filter = imu_transforms.IMUFilter(p=0.8, cut_off_frequency=6, sampling_frequency=25)
        imu_multi_person = imu_transforms.IMUMultiPerson(p=0.4, scale=[0.8, 1.2])
        imu_malfunction = imu_transforms.IMUMalFunction(p=0.1, mal_length=25)

        imu_toTensor = imu_transforms.ToTensor()

        data_transforms = transforms.Compose([imu_scale,
                                              # imu_error_model,  # mine
                                              imu_rotate,
                                              imu_negate,
                                              imu_flip,
                                              imu_warp,
                                              # imu_multi_person,    # mine
                                              # imu_malfunction,  # mine
                                              # imu_filter,    # mine
                                              imu_noise,
                                              imu_toTensor])
        return data_transforms

    def get_dataset(self, split, n_views=2, percent=20, shot=None):
        if self.transfer is False:
            # here it is for generating positive samples and negative samples
            return Dataset4Training(self.datasets_name, self.version, n_views, 
                                    transform=ContrastiveLearningViewGenerator(
                                        self.get_simclr_pipeline_transform(),
                                        n_views),
                                    split=split, transfer=self.transfer, percent=percent, shot=shot)
        elif split == 'train' or split == 'tune':
                # here it is for data augmentation, make it more challenging.
            return Dataset4Training(self.datasets_name, self.version, n_views,
                                    self.get_simclr_pipeline_transform(),
                                    split=split, transfer=self.transfer, percent=percent, shot=shot)
        else:  # val or test
            return Dataset4Training(self.datasets_name, self.version, n_views,
                                    transform=transforms.Compose([
                                        imu_transforms.ToTensor()
                                        ]),
                                    split=split, transfer=self.transfer, percent=percent, shot=shot)


class Dataset4Training(Dataset):

    def __init__(self, datasets_name, version, n_views=2, transform=None, split='train', transfer=True, percent=20, shot=None):
        """
        Args:
            datasets_name (string): dataset name.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            n_views: the defined views. defalt=2
            split: to specify train set or test set.
        """
        root_dir = fetch_dataset_root(datasets_name, version)

        train_dir = root_dir + '/train_set.npz'
        val_dir = root_dir + '/val_set.npz'
        test_dir = root_dir + '/test_set.npz'

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
                tune_dir = root_dir + '/tune_set_' + str(int(shot)) + '.npz'
            else:
                if percent <= 0.99:
                    tune_dir = root_dir + '/tune_set_' + str(percent).replace('.', '_') + '.npz'
                else:
                    tune_dir = root_dir + '/tune_set_' + str(int(percent)) + '.npz'
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

        sensor, label = self.sperate_label_data(sample)

        if self.transform:
            sensor = self.transform(sensor)
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
        elif self.datasets_name == 'ICHAR':
            label = np.array([int(add_infor[0, -1]), int(add_infor[0, -2]), int(add_infor[0, -3])])  # [movement, users, devices_type]
        elif self.datasets_name == 'HASC':
            label = np.array([int(add_infor[0, -1]), int(add_infor[0, 0]), int(add_infor[0, 1])])  # [movement, users, devices_type]
        else:
            raise InvalidDatasetSelection()
        sensor = np.concatenate((acc, gyro), axis=1)
        return sensor, torch.from_numpy(label).long()


class GenerativeDataset():
    def __init__(self, pipeline, version, datasets_name=None):
        self.pipeline = pipeline
        self.version = version
        self.datasets_name = datasets_name
    
    def get_dataset(self, split, percent=20):
        root = fetch_dataset_root(self.datasets_name, self.version)
        return LIMUDataset4Training(root, transform=self.pipeline, split=split, percent=percent)
############# mind the data transformation part or the pipeline part############


class LIMUDataset4Training(Dataset):

    def __init__(self, root_dir, transform=None, split='train', percent=20):
        """
        Args:
            root_dir (string): Path to all the npz file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            n_views: the defined views. defalt=2
            split: to specify train set or test set.
        """

        train_dir = '../../' + root_dir + '/train_set.npz'
        val_dir = '../../' + root_dir + '/val_set.npz'
        test_dir = '../../' + root_dir + '/test_set.npz'
        self.split = split
        if self.split == 'train':
            data = np.load(train_dir)
            self.windows_frame = data['train_set']
        elif self.split == 'val':
            data = np.load(val_dir)
            self.windows_frame = data['val_set']
        elif self.split == 'tune':
            if percent < 0.99:
                tune_dir = '../../' + root_dir + '/tune_set_' + str(percent).replace('.', '_') + '.npz'
            else:
                tune_dir = '../../' + root_dir + '/tune_set_' + str(int(percent)) + '.npz'
            data = np.load(tune_dir)
            self.windows_frame = data['tune_set']
        else:
            data = np.load(test_dir)
            self.windows_frame = data['test_set']

        self.split = split
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

        if self.transform:
            sample = self.transform(sample)
        return sample



# class MotionSense(Dataset):
#     """ MotionSense dataset, Huatao preprocessing way """

#     def __init__(self, root_dir, n_views=2, transform=None, split='train', transfer=True, percent=20):
#         """
#         Args:
#             root_dir (string): Path to all the npz file.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#             n_views: the defined views. defalt=2
#             split: to specify train set or test set.
#         """

#         train_dir = r'./datasets/MotionSense/train_set.npz'
#         val_dir = r'./datasets/MotionSense/val_set.npz'
#         test_dir = r'./datasets/MotionSense/test_set.npz'

#         self.transfer = transfer
#         self.split = split
#         if self.split == 'train':
#             data = np.load(train_dir)
#             self.windows_frame = data['train_set']
#         elif self.split == 'val':
#             data = np.load(val_dir)
#             self.windows_frame = data['val_set']
#         elif self.split == 'tune':
#             if percent <= 0.99:
#                 tune_dir = r'./datasets/MotionSense/tune_set_' + str(percent).replace('.', '_') + '.npz'
#             else:
#                 tune_dir = r'./datasets/MotionSense/tune_set_' + str(int(percent)) + '.npz'
#             data = np.load(tune_dir)
#             self.windows_frame = data['tune_set']
#         else:
#             data = np.load(test_dir)
#             self.windows_frame = data['test_set']

#         self.root_dir = root_dir
#         self.transform = transform
#         self.n_views = n_views

#     def __len__(self):
#         return len(self.windows_frame)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         loc = os.path.join(self.root_dir,
#                            self.windows_frame[idx])

#         sample = np.load(loc, allow_pickle=True)

#         if self.transform:
#             sample = self.transform(sample)

#         if not self.transfer:
#             sensor = [sample[i]['sensor'] for i in range(self.n_views)]
#             return sensor, sample[0]['label']
#         else:
#             sensor = sample['sensor']
#             return sensor, sample['label']
#         # return sample


# class UCI(Dataset):
#     """ UCI dataset, Huatao preprocessing way """

#     def __init__(self, root_dir, n_views=2, transform=None, split='train', transfer=True, percent=20):
#         """
#         Args:
#             root_dir (string): Path to all the npz file.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#             n_views: the defined views. defalt=2
#             split: to specify train set or test set.
#         """

#         train_dir = r'./datasets/UCI/train_set.npz'
#         val_dir = r'./datasets/UCI/val_set.npz'
#         test_dir = r'./datasets/UCI/test_set.npz'

#         self.transfer = transfer
#         self.split = split
#         if self.split == 'train':
#             data = np.load(train_dir)
#             self.windows_frame = data['train_set']
#         elif self.split == 'val':
#             data = np.load(val_dir)
#             self.windows_frame = data['val_set']
#         elif self.split == 'tune':
#             if percent <= 0.99:
#                 tune_dir = r'./datasets/UCI/tune_set_' + str(percent).replace('.', '_') + '.npz'
#             else:
#                 tune_dir = r'./datasets/UCI/tune_set_' + str(int(percent)) + '.npz'
#             data = np.load(tune_dir)
#             self.windows_frame = data['tune_set']
#         else:
#             data = np.load(test_dir)
#             self.windows_frame = data['test_set']

#         self.root_dir = root_dir
#         self.transform = transform
#         self.n_views = n_views

#     def __len__(self):
#         return len(self.windows_frame)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         loc = os.path.join(self.root_dir,
#                            self.windows_frame[idx])

#         sample = np.load(loc, allow_pickle=True)

#         if self.transform:
#             sample = self.transform(sample)

#         if not self.transfer:
#             sensor = [sample[i]['sensor'] for i in range(self.n_views)]
#             return sensor, sample[0]['label']
#         else:
#             sensor = sample['sensor']
#             return sensor, sample['label']
#         # return sample


# class Shoaib(Dataset):
#     """ Shoaib dataset, Huatao preprocessing way """

#     def __init__(self, root_dir, n_views=2, transform=None, split='train', transfer=True, percent=20):
#         """
#         Args:
#             root_dir (string): Path to all the npz file.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#             n_views: the defined views. defalt=2
#             split: to specify train set or test set.
#         """

#         train_dir = r'./datasets/Shoaib/train_set.npz'
#         val_dir = r'./datasets/Shoaib/val_set.npz'
#         test_dir = r'./datasets/Shoaib/test_set.npz'

#         self.transfer = transfer
#         self.split = split
#         if self.split == 'train':
#             data = np.load(train_dir)
#             self.windows_frame = data['train_set']
#         elif self.split == 'val':
#             data = np.load(val_dir)
#             self.windows_frame = data['val_set']
#         elif self.split == 'tune':
#             if percent <= 0.99:
#                 tune_dir = r'./datasets/Shoaib/tune_set_' + str(percent).replace('.', '_') + '.npz'
#             else:
#                 tune_dir = r'./datasets/Shoaib/tune_set_' + str(int(percent)) + '.npz'
#             data = np.load(tune_dir)
#             self.windows_frame = data['tune_set']
#         else:
#             data = np.load(test_dir)
#             self.windows_frame = data['test_set']

#         self.root_dir = root_dir
#         self.transform = transform
#         self.n_views = n_views

#     def __len__(self):
#         return len(self.windows_frame)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         loc = os.path.join(self.root_dir,
#                            self.windows_frame[idx])

#         sample = np.load(loc, allow_pickle=True)

#         if self.transform:
#             sample = self.transform(sample)

#         if not self.transfer:
#             sensor = [sample[i]['sensor'] for i in range(self.n_views)]
#             return sensor, sample[0]['label']
#         else:
#             sensor = sample['sensor']
#             return sensor, sample['label']
#         # return sample


# class UotHarDataset(Dataset):
#     """ UoT HAR dataset """

#     def __init__(self, root_dir, n_views=2, transform=None, split='train', transfer=True, percent=20):
#         """
#         Args:
#             root_dir (string): Path to all the npz file.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#             n_views: the defined views. defalt=2
#             split: to specify train set or test set.
#         """

#         train_dir = r'./datasets/uot_processed/train_set.npz'
#         val_dir = r'./datasets/uot_processed/val_set.npz'
#         test_dir = r'./datasets/uot_processed/test_set.npz'

#         self.transfer = transfer
#         self.split = split
#         if self.split == 'train':
#             data = np.load(train_dir)
#             self.windows_frame = data['train_set']
#         elif self.split == 'val':
#             data = np.load(val_dir)
#             self.windows_frame = data['val_set']
#         elif self.split == 'tune':
#             tune_dir = r'./datasets/uot_processed/tune_set_' + str(percent).replace('.', '_') + '.npz'
#             data = np.load(tune_dir)
#             self.windows_frame = data['tune_set']
#         else:
#             data = np.load(test_dir)
#             self.windows_frame = data['test_set']

#         self.root_dir = root_dir
#         self.transform = transform
#         self.n_views = n_views

#     def __len__(self):
#         return len(self.windows_frame)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         loc = os.path.join(self.root_dir,
#                            self.windows_frame[idx])

#         sample = np.load(loc, allow_pickle=True)

#         if self.transform:
#             sample = self.transform(sample)

#         if not self.transfer:
#             sensor = [sample[i]['sensor'] for i in range(self.n_views)]
#             return sensor, sample[0]['label']
#         else:
#             sensor = sample['sensor']
#             return sensor, sample['label']



# class UciHarDataset(Dataset):
#     """ UCI HAR dataset """

#     def __init__(self, root_dir, n_views=2, transform=None, split='train', transfer=True, percent=20):
#         """
#         Args:
#             root_dir (string): Path to all the npz file.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#             n_views: the defined views. defalt=2
#             split: to specify train set or test set.
#         """

#         train_dir = r'../datasets/processed/train_set.npz'
#         val_dir = r'./datasets/processed/val_set.npz'
#         test_dir = r'./datasets/processed/test_set.npz'

#         self.transfer = transfer
#         self.split = split
#         if self.split == 'train':
#             data = np.load(train_dir)
#             self.windows_frame = data['train_set']
#         elif self.split == 'val':
#             data = np.load(val_dir)
#             self.windows_frame = data['val_set']
#         elif self.split == 'tune':
#             tune_dir = r'./datasets/processed/tune_set_' + str(percent) + '.npz'
#             data = np.load(tune_dir)
#             self.windows_frame = data['tune_set']
#         else:
#             data = np.load(test_dir)
#             self.windows_frame = data['test_set']

#         self.root_dir = root_dir
#         self.transform = transform
#         self.n_views = n_views

#     def __len__(self):
#         return len(self.windows_frame)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         loc = os.path.join(self.root_dir,
#                            self.windows_frame[idx])

#         sample = np.load(loc, allow_pickle=True)

#         # if self.transform:
#         #     sample = self.transform(sample)
#         #
#         # if not self.transfer:
#         #     sensor = [sample[i]['sensor'] for i in range(self.n_views)]
#         #     return sensor, sample[0]['label']
#         # else:
#         #     sensor = sample['sensor']
#         #     return sensor, sample['label']
#         return sample


if __name__ == '__main__':
    root_folder = 'E:\\PyCharm 2020.3.3\\Project\\SimCLR\\datasets\\processed'
    list_folder = 'E:\\PyCharm 2020.3.3\\Project\\SimCLR\\datasets\\processed\\train_list.npz'
    datasets_name = 'UciHarDataset'
    n_views = 2
    batch_size = 2
    dataset = ContrastiveLearningDataset(root_folder, transfer=True)
    train_dataset = dataset.get_dataset(datasets_name, n_views, 'train')
    sample = train_dataset[12000]

    imu_discard = imu_transforms.IMUDiscard(datasets_name=datasets_name)

    imu_error_model = imu_transforms.IMUErrorModel(p=1, scale=[0.9, 1.1], error_magn=0.01, bias_magn=0.01)
    imu_filter = imu_transforms.IMUFilter(p=1, cut_off_frequency=6, sampling_frequency=25)
    imu_multi_person = imu_transforms.IMUMultiPerson(p=1, scale=[0.8, 1.2])
    imu_malfunction = imu_transforms.IMUMalFunction(p=1, mal_length=30)

    imu_noise = imu_transforms.IMUNoise(var=0.1, p=1)
    imu_scale = imu_transforms.IMUScale(scale=[0.9, 1.1], p=1)
    imu_rotate = imu_transforms.IMURotate(p=1)

    imu_negate = imu_transforms.IMUNegated(p=1)
    imu_flip = imu_transforms.IMUHorizontalFlip(p=1)
    imu_warp = imu_transforms.IMUTimeWarp(p=1)

    sample = imu_discard(sample)
    y = imu_error_model(sample)
    figure_cmp_transformation(sample['acc'], y['acc'], 'Original', type(imu_error_model).__name__)

    y = imu_filter(sample)
    figure_cmp_transformation(sample['acc'], y['acc'], 'Original', type(imu_filter).__name__)

    y = imu_negate(sample)
    figure_cmp_transformation(sample['acc'], y['acc'], 'Original', type(imu_negate).__name__)

    y = imu_malfunction(sample)
    figure_cmp_transformation(sample['acc'], y['acc'], 'Original', type(imu_malfunction).__name__)

    y = imu_rotate(sample)
    figure_cmp_transformation(sample['acc'], y['acc'], 'Original', type(imu_rotate).__name__)

    y = imu_warp(sample)
    figure_cmp_transformation(sample['acc'], y['acc'], 'Original', type(imu_warp).__name__)

    y = imu_multi_person(sample)
    figure_cmp_transformation(sample['acc'], y['acc'], 'Original', type(imu_multi_person).__name__)
    # ax = plt.subplot(4, 1, 1)
    # plt.tight_layout()
    # plt.plot(sample['acc'])
    # ax.set_title('Original')
    # sample = imu_discard(sample)
    # for i, tsfrm in enumerate([imu_noise, imu_scale, imu_rotate]):
    #     sample = tsfrm(sample)
    #     ax = plt.subplot(4, 1, i + 2)
    #     plt.tight_layout()
    #     plt.plot(sample['acc'])
    #     ax.set_title(type(tsfrm).__name__)
    # plt.show()
    #
    # plt.figure()
    # ax = plt.subplot(4, 1, 1)
    # plt.tight_layout()
    # plt.plot(sample['acc'])
    # ax.set_title('IMURotate')
    # for i, tsfrm in enumerate([imu_negate, imu_flip, imu_warp]):
    #     sample = tsfrm(sample)
    #     ax = plt.subplot(4, 1, i + 2)
    #     plt.tight_layout()
    #     plt.plot(sample['acc'])
    #     ax.set_title(type(tsfrm).__name__)
    # plt.show()
    #
    # plt.figure()
    # ax = plt.subplot(5, 1, 1)
    # plt.tight_layout()
    # plt.plot(sample['acc'])
    # ax.set_title('IMUWarp')
    # for i, tsfrm in enumerate([imu_error_model, imu_filter, imu_multi_person, imu_malfunction]):
    #     sample = tsfrm(sample)
    #     ax = plt.subplot(5, 1, i + 2)
    #     plt.tight_layout()
    #     plt.plot(sample['acc'])
    #     ax.set_title(type(tsfrm).__name__)
    # plt.show()

    # for i in range(len(train_dataset)):
    #     sample = train_dataset[i]
    #     plt.figure_plot(sample['acc'])
    #     plt.show()
    #     break

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True,
    #     num_workers=4, pin_memory=True, drop_last=True)
    #
    # for sensor, _ in train_loader:
    #     print(sensor)
    #     sensor = torch.cat(sensor, dim=0)
    #     print(sensor)
    #     break
