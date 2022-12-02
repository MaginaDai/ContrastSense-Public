import argparse
from ast import parse
from itertools import dropwhile
from math import gamma
import os
import pdb
import string

import torch
import torch.backends.cudnn as cudnn
from torchvision import models

from CPC import CPCV1, CPC
from MoCo import MoCo_v1, MoCo
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from data_aug.preprocessing import UsersNum
from simclr import SimCLR, MyNet, LIMU_encoder
from utils import load_model_config, seed_torch
import numpy as np
import random
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning for Wearable Sensing')

parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--transfer', default=False, type=str, help='to tell whether we are doing transfer learning')
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')
parser.add_argument('--resume', default='', type=str, help='To restart the model from a previous model')

parser.add_argument('--out_dim', default=512, type=int,
                    help='feature dimension (default: 512)')
parser.add_argument('-t', '--temperature', default=0.1, type=float,
                    help='softmax temperature (default: 1)')
parser.add_argument('--store', default='test_HHAR', type=str, help='define the name head for model storing')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-name', default='HASC',
                    help='datasets name', choices=['HHAR', 'MotionSense', 'UCI', 'Shoaib', 'ICHAR', 'HASC'])
parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 5)')
parser.add_argument('-e', '--epochs', default=2000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--log-every-n-steps', default=5, type=int, help='Log every n steps')

parser.add_argument('-g', '--gpu-index', default=1, type=int, help='Gpu index.')
parser.add_argument('--best-acc', default=0., type=float, help='The initial best accuracy')
parser.add_argument('-mol', default='MoCo', type=str, help='which model to use', choices=['SimCLR', 'LIMU', 'CPC', 'MoCo', 'DeepSense'])
parser.add_argument('--timestep', default=15, type=int, help='how many time steps for CPC')
parser.add_argument('-d', default=32, type=int, help='how dim for CPC')
parser.add_argument('-moco_K', default=1024, type=int, help='keys size')
parser.add_argument('-moco_m', default=0.999, type=float, help='momentum value')

parser.add_argument('-eta_min', default=1e-5, type=float, help='the lowest lr')
parser.add_argument('-T_max_ratio', default=0.8, type=float, help='the stop-lr-tuning stage')
parser.add_argument('-s_step', default=500, type=int, help='the step size of StepLR')
parser.add_argument('-s_gamma', default=0.5, type=float, help='the gamma of StepLR')

parser.add_argument('-label_type', default=1, type=int, help='How many different kinds of labels for pretraining')
parser.add_argument('-slr', default=[0.3], nargs='+', type=float, help='the ratio of sup_loss')
parser.add_argument('-tem_labels', default=[0.1], nargs='+', type=float, help='the temperature for supervised CL')

parser.add_argument('-num_clusters', default=None, type=int, help='number of clusters for K-means')
parser.add_argument('-iter_tol', default=None, type=float, help='Max iteration number for clustering')

parser.add_argument('-final_dim', default=8, type=int, help='the output dims of the GRU')
parser.add_argument('-mo', default=0.9, type=float, help='the momentum for Batch Normalization')

parser.add_argument('-drop', default=0.1, type=float, help='the dropout portion')
parser.add_argument('-version', default="shot", type=str, help='control the version of the setting')
parser.add_argument('-DAL', default=True, type=bool, help='Use Domain Adaversarial Learning or not')
parser.add_argument('-CE', default=False, type=bool, help='Use Cross Entropy Domain Loss or not')

fishermax=0.0001

def getFisherDiagonal(args):
    args.label_type = 1
    args.num_clusters = None
    args.iter_tol = None

    dataset = ContrastiveLearningDataset(transfer=False, version=args.version, datasets_name=args.name)

    train_dataset = dataset.get_dataset(split='train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=10, pin_memory=False, drop_last=True)

    user_num = None
    model = MoCo_v1(device=args.device, mol=args.mol)
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint['state_dict']
            log = model.load_state_dict(state_dict, strict=False)
    else:
        raise NotADirectoryError

    fisher = {
            n[len("encoder_q."):]: torch.zeros(p.shape).to(args.device)
            for n, p in model.named_parameters()
            if p.requires_grad and n.startswith('encoder_q.encoder')
        }
    
    model.to(args.device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-4)

    for sensor, labels in train_loader:
        sensor = [t.to(args.device) for t in sensor]
        gt_label = labels[:, 0].to(args.device) # the first dim is motion labels
        sup_label = [labels[:, i + 1].to(args.device) for i in range(args.label_type)]  # the following dim are cheap labels
        output, target, logits_labels, cluster_eval, cluster_loss, center_shift, feature = model(sensor[0], sensor[1], labels=sup_label, 
                                                                                            num_clusters=args.num_clusters, 
                                                                                            iter_tol=args.iter_tol,
                                                                                            gt=gt_label, if_plot=False,
                                                                                            n_iter=0)
        sup_loss = model.supervised_CL(logits_labels=logits_labels, labels=sup_label)
        loss = - sup_loss
        optimizer.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None and n.startswith('encoder_q.encoder'):
                fisher[n[len("encoder_q."):]] += p.grad.pow(2).clone()
    for n, p in fisher.items():
        fisher[n] = p / len(train_loader)
        fisher[n] = torch.min(fisher[n], torch.tensor(args.fishermax)).to(args.device)
    return fisher


if __name__ == '__main__':
    args = parser.parse_args()
    fisher = getFisherDiagonal(args)