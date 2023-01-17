import torch
import sys
import numpy as np
import os
from os.path import dirname
sys.path.append(dirname(dirname(sys.path[0])))

from torchvision.transforms import transforms
from torch.utils.data import Dataset
from data_aug.contrastive_learning_dataset import fetch_dataset_root
from data_aug.preprocessing import UsersPosition
from data_aug.view_generator import ContrastiveLearningViewGenerator, SingleViewGenerator
from data_aug import imu_transforms
from exceptions.exceptions import InvalidDatasetSelection


class CL_Rotate(object):
    """ Rotate the IMU sensors reading based on CLHAR design

    Args:
        p: transformation possibility
    """

    def __init__(self):
        
        return

    def __call__(self, sensor, pos=None):
        acc, gyro = sensor[:, 0:3], sensor[:, 3:]

        axes = np.random.uniform(low=-1, high=1, size=(1, 3))  # we apply to each sensor rather than a batch size. 
        angles = np.random.uniform(low=-np.pi, high=np.pi, size=(1))
        matrices = axis_angle_to_rotation_matrix_3d_vectorized(axes, angles)[0]

        acc = np.dot(acc, matrices)
        gyro = np.dot(gyro, matrices)
        sensor = np.concatenate((acc, gyro), axis=-1)
        return sensor

def axis_angle_to_rotation_matrix_3d_vectorized(axes, angles):
    """
    Copy from CLHAR
    """
    axes = axes / np.linalg.norm(axes, ord=2, axis=1, keepdims=True)
    x = axes[:, 0]; y = axes[:, 1]; z = axes[:, 2]
    c = np.cos(angles)
    s = np.sin(angles)
    C = 1 - c

    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC

    m = np.array([
        [ x*xC+c,   xyC-zs,   zxC+ys ],
        [ xyC+zs,   y*yC+c,   yzC-xs ],
        [ zxC-ys,   yzC+xs,   z*zC+c ]])
    matrix_transposed = np.transpose(m, axes=(2,0,1))
    return matrix_transposed


class CLHAR_Dataset:
    def __init__(self, transfer, version, datasets_name=None):
        self.transfer = transfer
        self.datasets_name = datasets_name
        self.version = version

    def get_simclr_pipeline_transform(self):
        """Return a set of data augmentation transformations as described in my presentation."""
        imu_toTensor = imu_transforms.ToTensor()
        imu_rotate = CL_Rotate()
        data_transforms = transforms.Compose([imu_rotate,
                                              imu_toTensor])
        return data_transforms

    def get_dataset(self, split, n_views=2, percent=20, shot=None):
        if self.transfer is False:
            # here it is for generating positive samples and negative samples
            return CLHARDataset4Training(self.datasets_name, self.version, n_views, 
                                    transform=ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(), n_views=n_views),
                                    split=split, transfer=self.transfer, percent=percent, shot=shot)
        elif split == 'train' or split == 'tune':
                # here it is for data augmentation, make it more challenging.
            return CLHARDataset4Training(self.datasets_name, self.version, n_views,
                                    self.get_simclr_pipeline_transform(),
                                    split=split, transfer=self.transfer, percent=percent, shot=shot)
        else:  # val or test
            return CLHARDataset4Training(self.datasets_name, self.version, n_views,
                                    transform=transforms.Compose([
                                        imu_transforms.ToTensor()
                                        ]),
                                    split=split, transfer=self.transfer, percent=percent, shot=shot)


class CLHARDataset4Training(Dataset):

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