import os
import sys
import numpy as np
import torch
from os.path import dirname
sys.path.append(dirname(dirname(sys.path[0])))

from baseline.CPCHAR.dataload import CPCHARDataset4Training, ToTensor
from baseline.CLHAR.CL_dataload import CL_Rotate
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from data_aug.imu_transforms import IMUTimeWarp

class FMUDA_Dataset:
    def __init__(self, transfer, version, datasets_name=None):
        self.transfer = transfer
        self.datasets_name = datasets_name
        self.version = version

    def get_simclr_pipeline_transform(self):
        """Return a set of data augmentation transformations as described in my presentation."""
        imu_toTensor = ToTensor()
        imu_rotate = CL_Rotate()
        imu_scale = IMUScale()
        imu_warp = IMUTimeWarp(p=1.0)
        data_transforms = transforms.Compose([imu_rotate,
                                              imu_scale,
                                              imu_warp,
                                              imu_toTensor])
        return data_transforms

    def get_dataset(self, split, percent=20, shot=None):
        if split == 'train' or split == 'tune':
            return CPCHARDataset4Training(self.datasets_name, self.version,
                                        transform=self.get_simclr_pipeline_transform(),  # can add more transformations
                                        split=split, transfer=self.transfer, percent=percent, shot=shot)
        else:
            return CPCHARDataset4Training(self.datasets_name, self.version,
                                        transform=transforms.Compose([ToTensor()]),
                                        split=split, transfer=self.transfer, percent=percent, shot=shot)
    

class IMUScale(object):
    """ Rescale the IMU sensors reading

    Args:
        scale: the scaling range for all axis
        p: transformation possibility
    """

    def __init__(self, mu=1, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sensor):
        # acc, gyro = sample['acc'], sample['gyro']
        acc, gyro = sensor[:, 0:3], sensor[:, 3:]
        K = np.eye(3)
        for i in range(3):
            K[i, i] *= np.random.normal(self.mu, self.sigma)
        acc = np.dot(acc, K)

        K = np.eye(3)
        for i in range(3):
            K[i, i] *= np.random.normal(self.mu, self.sigma)
        gyro = np.dot(gyro, K)
        sensor = np.concatenate((acc, gyro), axis=1)
        return sensor