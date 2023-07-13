
from SACL.dataloader import SADataset

import numpy as np


def load_CLISA_data(name, version):
    dataset = SADataset(transfer=False, version=version, datasets_name=name)
    train_dataset = dataset.get_dataset(split='train')

    tune_domain_loader = []
    
    domain_types = np.unique(domain_label)

    return

