import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import sys
from os.path import dirname
sys.path.append(dirname(dirname(sys.path[0])))

from data_aug.contrastive_learning_dataset import fetch_dataset_root
from data_aug.preprocessing import UsersPosition
from exceptions.exceptions import InvalidDatasetSelection

def get_sample_weights(y, weights):
    '''
    to assign weights to each sample
    '''
    label_unique = np.unique(y)
    sample_weights = []
    for val in y:
        idx = np.where(label_unique == val)
        sample_weights.append(weights[idx])
    return sample_weights


def sperate_label_data(sample, datasets_name):
        acc, gyro, label = sample['acc'], sample['gyro'], sample['add_infor']
        acc /= 9.8
        sensor = np.concatenate((acc, gyro), axis=1)
        return sensor, label


def load_data(datasets_name, version, split, cross, shot=None):
    root_dir = fetch_dataset_root(datasets_name)

    train_dir = '../../' + root_dir + '_' + version + '/train_set.npz'
    val_dir = '../../' + root_dir + '_' + version + '/val_set.npz'
    test_dir = '../../' + root_dir + '_' + version + '/test_set.npz'
    tune_dir = '../../' + root_dir + '_' + version + '/tune_set_' + str(int(shot)) + '.npz'

    if split == 'train':
        data = np.load(train_dir)
        windows_frame = data['train_set']
    elif split == 'val':
        data = np.load(val_dir)
        windows_frame = data['val_set']
    elif split == 'tune':
        data = np.load(tune_dir)
        windows_frame = data['tune_set']
    else:
        data = np.load(test_dir)
        windows_frame = data['test_set']

    X = np.zeros([len(windows_frame), 200, 6])
    y = np.zeros(len(windows_frame))
    d = np.zeros(len(windows_frame))

    for idx, file_name in enumerate(windows_frame):
        loc = os.path.join('../../' + root_dir, file_name)

        sample = np.load(loc, allow_pickle=True)

        sensor, label = sperate_label_data(sample, datasets_name)
        
        X[idx] = sensor
        y[idx] = label[0]
        if cross == 'users':
            d[idx] = label[1]
        elif cross == 'positions' or cross == 'devices':
            d[idx] = label[2]
        elif cross == 'multiple':
            d[idx] = label[3]
        else:
            NotADirectoryError

    return X, y, d    


def load_GILE_type_data(datasets_name, version, shot, batch_size, setting, cross):
    tune_domain_loader = []

    if setting == 'full':  # load fully labeled data in the training set
        data, motion_label, domain_label = load_data(datasets_name, version, 'train', cross, shot)
    else:
        data, motion_label, domain_label = load_data(datasets_name, version, 'tune', cross, shot)
    domain_types = np.unique(domain_label)

    for domain_type in domain_types:
        idx = domain_label == domain_type
        x = data[idx]
        y = motion_label[idx]
        d = domain_label[idx]

        x = np.transpose(x.reshape((-1, 1, 200, 6)), (0, 2, 1, 3))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0, 0, 0, 0], std=[1, 1, 1, 1, 1, 1])
        ])
        domain_set = data_loader(x, y, d, transform)
        unique_y, counts_y = np.unique(y, return_counts=True)
        weights = 100.0 / torch.Tensor(counts_y)
        weights = weights.double()

        sample_weights = get_sample_weights(y, weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        source_loader = DataLoader(domain_set, batch_size=batch_size, shuffle=False, drop_last=False, sampler=sampler)
        
        tune_domain_loader.append(source_loader)

    val_data, val_motion_label, val_domain_label = load_data(datasets_name, version, 'val', cross, shot)

    val_data = np.transpose(val_data.reshape((-1, 1, 200, 6)), (0, 2, 1, 3))
    val_set = data_loader(val_data, val_motion_label, val_domain_label, transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    test_data, test_motion_label, test_domain_label = load_data(datasets_name, version, 'test', cross, shot)
    test_data = np.transpose(test_data.reshape((-1, 1, 200, 6)), (0, 2, 1, 3))
    test_set = data_loader(test_data, test_motion_label, test_domain_label, transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return tune_domain_loader, val_loader, test_loader


class data_loader(Dataset):
    def __init__(self, samples, labels, domains, t):
        self.samples = samples
        self.labels = labels
        self.domains = domains
        self.T = t

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        # means = torch.mean(sample, dim=0)
        # stds = torch.std(sample, dim=0)
        # sample = (sample - means) / stds
        sample = self.T(sample)
        return np.transpose(sample, (1, 0, 2)), target.astype(int), domain.astype(int)

    def __len__(self):
        return len(self.samples)


