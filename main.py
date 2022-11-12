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


def main():
    args = parser.parse_args()
    # check if gpu training is available

    seed_torch(seed=args.seed)

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_index}')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    if args.mol == 'CPC':  # set transfer = True to avoid n_views = 2. Just to make the dataset normal
        dataset = ContrastiveLearningDataset(transfer=True, version=args.version, datasets_name=args.name)
    else:
        dataset = ContrastiveLearningDataset(transfer=False, version=args.version, datasets_name=args.name)

    train_dataset = dataset.get_dataset(split='train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    if args.mol == 'LIMU':
        model_cfg = load_model_config(target='pretrain_base', prefix='base', version='v1')
        model = LIMU_encoder(model_cfg)
    elif args.mol == 'CPC':
        model = CPCV1(timestep=args.timestep, batch_size=args.batch_size, seq_len=96, transfer=False, classes=6, dims=args.d, temperature=args.temperature)
    elif args.mol == 'MoCo' or args.mol == 'DeepSense':
        if args.CE:
            user_num = UsersNum[args.name]
        else:
            user_num = None
        model = MoCo_v1(device=args.device, out_dim=args.out_dim, K=args.moco_K, m=args.moco_m, T=args.temperature, 
                        T_labels=args.tem_labels, dims=args.d, label_type=args.label_type, num_clusters=args.num_clusters, mol=args.mol, 
                        final_dim=args.final_dim, momentum=args.mo, drop=args.drop, DAL=args.DAL, if_cross_entropy=args.CE, users_class=user_num)
    else:
        model = MyNet(transfer=False, out_dim=args.out_dim, if_bn=args.if_bn, if_g=args.if_g, if_lstm=args.if_lstm)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs/2)], gamma=args.s_gamma, last_epoch=-1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            args.start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            best_acc = best_acc.to(args.device)  # best_acc1 may be from a checkpoint from a different GPU
            args.best_acc = best_acc
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        if args.mol == 'CPC':
            cpc = CPC(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
            cpc.train(train_loader)
        elif args.mol == 'MoCo' or args.mol == 'DeepSense':
            moco = MoCo(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
            moco.train(train_loader)
        else:
            simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
            simclr.train(train_loader)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
