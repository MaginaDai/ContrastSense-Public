import argparse
import sys
import torch
from os.path import dirname
import numpy as np

sys.path.append(dirname(dirname(sys.path[0])))
sys.path.append(dirname(sys.path[0]))

from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from data_aug.preprocessing import ClassesNum, UsersNum
from SACL import SACL
from dataloader import SADataset
from model import SACL_model, SACLAdversary
from utils import seed_torch

parser = argparse.ArgumentParser(description='PyTorch Contrastive Domain Adaptation')

parser.add_argument('-lr', '--learning-rate', default=5e-4, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N')
parser.add_argument('-e', '--epochs', default=1000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-t', '--temperature', default=0.5, type=float, help='softmax temperature (default: 0.1)')
parser.add_argument('-name', default='SEED', help='EEG datasets name', choices=['SEED', 'SEED_IV'])

parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('--resume', default='', type=str, help='To restart the model from a previous model')
parser.add_argument('--transfer', default=False, type=str, help='to tell whether we are doing transfer learning')
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')
parser.add_argument('--log-every-n-steps', default=5, type=int, help='Log every n steps')
parser.add_argument('-g', '--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 5)')
parser.add_argument('--store', default='lr', type=str, help='define the name head for model storing')
parser.add_argument('-version', default="shot0", type=str, help='control the version of the setting')

def main():
    args = parser.parse_args()
    seed_torch(seed=args.seed)

    args.name = args.name
    args.modal = 'eeg'

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_index}')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset(transfer=False, version=args.version, datasets_name=args.name, if_baseline=True, modal='eeg')

    train_dataset = dataset.get_dataset(split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=False, drop_last=True)
    
    # val_dataset = dataset.get_dataset(split='val')
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
    #                 num_workers=args.workers, pin_memory=False, drop_last=True)
    num_subject, map_subject_to_idx = mapping_subject_id(train_loader)

    model = SACL_model(device=args.device, num_subjects=num_subject)
    
    optimizer = torch.optim.Adam(model.model.parameters(), lr=args.lr)
    adversarial_optimizer = torch.optim.Adam(model.adversary.parameters(), lr=args.lr)
    
    with torch.cuda.device(args.gpu_index):
        sacl = SACL(model=model, optimizer=optimizer, adversarial_optimizer=adversarial_optimizer, args=args)
        sacl.train(train_loader, map_subject_to_idx)
    return

def mapping_subject_id(train_loader):
    subject_id = torch.empty(0)
    for _, label in train_loader:
        domain_label = label[:, 1]
        subject_id = torch.cat([domain_label, subject_id])
    
    subject_id = torch.unique(subject_id).long()
    num_subject = len(subject_id)
    map_subject_to_idx = (-1) * torch.ones(15).long()
    map_subject_to_idx[subject_id] = torch.arange(num_subject).long()
    return num_subject, map_subject_to_idx

    


if __name__ == '__main__':
    main()