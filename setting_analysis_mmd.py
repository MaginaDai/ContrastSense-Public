
import torch
import argparse
from torch import nn
from torchvision.transforms import transforms
from data_aug import emg_transforms, imu_transforms
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, Dataset4Training
from utils import seed_torch
import numpy as np
import pdb

parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning for Wearable Sensing')

parser.add_argument('-name', default='HHAR', help='datasets name', choices=['HHAR', 'MotionSense', 'Shoaib', 'HASC', 'Myo', 'NinaPro', 'sleepEDF', "Merged_dataset"])
parser.add_argument('-version', default="HASC", type=str, help='control the version of the setting')
parser.add_argument('-cross', default='users', type=str, help='decide to use which kind of labels')
parser.add_argument('--store', default='test', type=str, help='define the name head for model storing')
parser.add_argument('-shot', default=10, type=int, help='how many shots of labels to use')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')


## based on the code from https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py
class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        # pdb.set_trace()
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        # pdb.set_trace()
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        # pdb.set_trace()
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
    


def analysis_the_mmd_distance(version, dataset, tune=False):
    args = parser.parse_args()
    # check if gpu training is available
    args.name = dataset
    seed_torch(seed=args.seed)

    # if torch.cuda.is_available():
    #     args.device = torch.device(f'cuda:{args.gpu_index}')
    # else:
    #     args.device = torch.device('cpu')
    #     args.gpu_index = -1


    if args.name in ['NinaPro', 'Myo', 'UCI']:
        args.modal = 'emg'
    elif args.name == 'sleepEDF':
        args.modal = 'eeg'
    else:
        args.modal = 'imu'
    
    # dataset = ContrastiveLearningDataset(transfer=False, version=version, datasets_name=args.name, modal=args.modal)

    # train_dataset = dataset.get_dataset(split='train')

    if tune is False:
        train_dataset = Dataset4Training(args.name, version, transform=transforms.Compose([imu_transforms.ToTensor()]),  split='train', transfer=True, modal=args.modal)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False, drop_last=True)
    elif tune is True:
        dataset = ContrastiveLearningDataset(transfer=True, version=version, datasets_name=args.name, modal=args.modal)
        if args.modal == 'imu':
            tune_dataset = Dataset4Training(args.name, version, transform=transforms.Compose([imu_transforms.ToTensor()]), 
                                            split='tune', transfer=True, shot=args.shot, modal=args.modal)
        elif args.modal == 'emg':
            tune_dataset = Dataset4Training(args.name, version, transform=transforms.Compose([emg_transforms.EMGToTensor()]), 
                                            split='tune', transfer=True, shot=args.shot, modal=args.modal)
        else:
            NotADirectoryError
        
        train_loader = torch.utils.data.DataLoader(tune_dataset, batch_size=int(args.batch_size), shuffle=True, pin_memory=False, drop_last=False)
    

    dataset = ContrastiveLearningDataset(transfer=True, version=version, datasets_name=args.name, modal=args.modal)
    test_dataset = dataset.get_dataset('test')

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False, drop_last=False)
    

    mmd_distance = MMDLoss()

    X=torch.empty(0)
    Y=torch.empty(0)
    for sensor, _ in train_loader:
        # sensor = sensor.to(args.device)
        sensor = sensor.reshape(sensor.shape[0], -1)
        if len(X) == 0:
            X = sensor
        else:
            X = torch.vstack((X, sensor))
        # pdb.set_trace()
    for sensor, _ in test_loader:
        # sensor = sensor.to(args.device)
        # pdb.set_trace()
        sensor = sensor.reshape(sensor.shape[0], -1)
        if len(Y) == 0:
            Y = sensor
        else:
            Y = torch.vstack((Y, sensor))
    
    # pdb.set_trace()
    distance = mmd_distance(X, Y)
    print(distance)
    return distance


def cross_user_scenario():
    version = ["shot"]
    # datasets = ["HHAR", "MotionSense", "Shoaib", "HASC"]
    datasets = ["Myo", "NinaPro", "Shoaib", "HASC"]
    distance = torch.zeros([len(datasets), 5])
    # for i, v in enumerate(version):
    #     for j in range(5):
    #         for m, dataset in enumerate(datasets):
    #             distance[m, j] = analysis_the_mmd_distance(v + str(j), dataset, tune=True)
    distance = torch.tensor([[0.0659, 0.0292, 0.0321, 0.0334, 0.0619],
        [0.0234, 0.0102, 0.0046, 0.0119, 0.0026],
        [0.0079, 0.0116, 0.0043, 0.0032, 0.0076],
        [0.1132, 0.0154, 0.0131, 0.0033, 0.0303]])

    print(distance)
    print(torch.mean(distance, dim=1))
    print(torch.std(distance, dim=1))

def cross_device_position():
    version = ["users_positions_shot"]
    datasets = ["Shoaib"]
    distance = torch.zeros([len(datasets), 5])
    for i, v in enumerate(version):
        for j in range(5):
            for m, dataset in enumerate(datasets):
                distance[m, j] = analysis_the_mmd_distance(v + str(j), dataset, tune=True)
    print(distance)
    print(torch.mean(distance, dim=1))
    print(torch.std(distance, dim=1))

def cross_dataset_scenario():
    version = ["HHAR", "HASC", "MotionSense", "Shoaib"]
    datasets = ["Merged_dataset"]
    distance = torch.zeros([len(version), 5])
    for i, v in enumerate(version):
        for j in range(5):
            for m, dataset in enumerate(datasets):
                distance[i, j] = analysis_the_mmd_distance(v + "_shot" +str(j), dataset, tune=False)
    print(distance)
    print(torch.mean(distance, dim=1))
    print(torch.std(distance, dim=1))

if __name__ == "__main__":
    cross_user_scenario()