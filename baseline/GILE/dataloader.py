import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def get_sample_weights(y, weights):
    '''
    to assign weights to each sample
    '''
    label_unique = torch.unique(y)
    sample_weights = []
    for val in y:
        idx = torch.where(label_unique == val)
        sample_weights.append(weights[idx])
    return sample_weights


def load_GILE_type_data(tune_dataset, val_dataset, test_dataset, batch_size):
    tune_domain_loader = []

    data = torch.zeros([len(tune_dataset), 200, 6])
    domain_label = torch.zeros(len(tune_dataset))
    motion_label = torch.zeros(len(tune_dataset))

    for idx, (sensor, target) in enumerate(tune_dataset):
        data[idx] = sensor
        motion_label[idx] = target[0]
        domain_label[idx] = target[1]
        
    domain_types = torch.unique(domain_label)

    for domain_type in domain_types:
        idx = domain_label == domain_type
        x = data[idx]
        y = motion_label[idx]
        d = domain_label[idx]

        x = torch.permute(x.reshape((-1, 1, 200, 6)), (0, 2, 1, 3))
        
        domain_set = data_loader(x, y, d)
        unique_y, counts_y = np.unique(y, return_counts=True)
        weights = 100.0 / torch.Tensor(counts_y)
        weights = weights.double()

        sample_weights = get_sample_weights(y, weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        source_loader = DataLoader(domain_set, batch_size=batch_size, shuffle=False, drop_last=False, sampler=sampler)
        
        tune_domain_loader.append(source_loader)
    
    val_data = torch.zeros([len(val_dataset), 200, 6])
    val_domain_label = torch.zeros(len(val_dataset))
    val_motion_label = torch.zeros(len(val_dataset))
    
    for idx, (sensor, target) in enumerate(val_dataset):
        val_data[idx] = sensor
        val_motion_label[idx] = target[0]
        val_domain_label[idx] = target[1]
        

    val_data = torch.permute(val_data.reshape((-1, 1, 200, 6)), (0, 2, 1, 3))
    val_set = data_loader(val_data, val_motion_label, val_domain_label)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    test_data = torch.zeros([len(test_dataset), 200, 6])
    test_domain_label = torch.zeros(len(test_dataset))
    test_motion_label = torch.zeros(len(test_dataset))
    
    for idx, (sensor, target) in enumerate(test_dataset):
        test_data[idx] = sensor
        test_motion_label[idx] = target[0]
        test_domain_label[idx] = target[1]

    test_data = torch.permute(test_data.reshape((-1, 1, 200, 6)), (0, 2, 1, 3))
    test_set = data_loader(test_data, test_motion_label, test_domain_label)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return tune_domain_loader, val_loader, test_loader


class data_loader(Dataset):
    def __init__(self, samples, labels, domains):
        self.samples = samples
        self.labels = labels
        self.domains = domains

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        means = torch.mean(sample, dim=0)
        stds = torch.std(sample, dim=0)
        sample = (sample - means) / stds
        return torch.permute(sample, [1, 0, 2]), target.long(), domain.long()

    def __len__(self):
        return len(self.samples)