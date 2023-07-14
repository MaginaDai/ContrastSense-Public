
import os, sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from os.path import dirname
sys.path.append(dirname(dirname(sys.path[0])))
sys.path.append(dirname(sys.path[0]))

from data_aug.contrastive_learning_dataset import fetch_dataset_root


def load_CLISA_data(datasets_name, version):
    tune_domain_loader = []
    data, class_label, domain_label, stimulus_label = load_data(datasets_name, version, 'train')
    
    domain_types = np.unique(domain_label)

    for domain_type in domain_types:
        idx = domain_label == domain_type
        x = data[idx]
        y = class_label[idx]
        d = domain_label[idx]
        stimulus=  stimulus_label[idx]
        stimulus_types = np.unique(stimulus)

        x_s = []
        y_s = []
        d_s = []
        s_s = []
        
        minimum_len = 1e4
        for j in stimulus_types:
            pos = stimulus == j
            x_s.append(x[pos])
            y_s.append(y[pos])
            d_s.append(d[pos])
            s_s.append(stimulus[pos])
            if len(x[pos]) < minimum_len:
                minimum_len = len(x[pos])

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        domain_set = data_loader(x_s, y_s, d_s, s_s, minimum_len, transform)

        source_loader = DataLoader(domain_set, batch_size=1, shuffle=True, drop_last=False)  # for each time. we sample 1 group of sample from each stimulus.
        
        tune_domain_loader.append(source_loader)

    return tune_domain_loader


class data_loader(Dataset):
    def __init__(self, samples, labels, domains, stimulus, minimum_len, t):
        self.samples = samples
        self.labels = labels
        self.domains = domains
        self.stimulus = stimulus
        self.minimum_len = minimum_len
        self.T = t

    def __getitem__(self, index):
        sample = []
        label = []
        domain = []
        stimulus = []
        
        for i in range(len(self.samples)):
            sample.append(self.T(self.samples[i][index].astype('float32')))
            label.append(self.labels[i][index])
            domain.append(self.domains[i][index])
            stimulus.append(self.stimulus[i][index])
        
        return torch.vstack(sample), torch.Tensor(label), torch.Tensor(domain), torch.Tensor(stimulus)

    def __len__(self):
        return self.minimum_len


def load_data(datasets_name, version, split, shot=None):
    root_dir = fetch_dataset_root(datasets_name)

    train_dir = '../../' + root_dir + '_' + version + '/train_set.npz'
    val_dir = '../../' + root_dir + '_' + version + '/val_set.npz'
    test_dir = '../../' + root_dir + '_' + version + '/test_set.npz'
    if shot==None:
        shot=0
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

    X = np.zeros([len(windows_frame), 200, 62])
    class_label = np.zeros(len(windows_frame))
    domain_label = np.zeros(len(windows_frame))
    stimulus_label = np.zeros(len(windows_frame))

    for idx, file_name in enumerate(windows_frame):
        loc = os.path.join('../../' + root_dir, file_name)

        sample = np.load(loc, allow_pickle=True)

        sensor, label = sample['eeg'], sample['add_infor']

        label = torch.from_numpy(label).long()
        
        X[idx] = sensor
        class_label[idx] = label[0]
        domain_label[idx] = label[1]
        stimulus_label[idx] = label[2]

    return X, class_label, domain_label, stimulus_label    


if __name__ == '__main__':
    train_loader = load_CLISA_data(name='SEED', version='shot0')