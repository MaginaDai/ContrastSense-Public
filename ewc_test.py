import argparse
from copy import deepcopy
import logging
import os
import pdb
import string

import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np

from torchvision import models

from CPC import CPCV1, CPC
from DeepSense import DeepSense_encoder
from MoCo import MoCo_model, MoCo_v1, MoCo_encoder, MoCo
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, Dataset4Training
from data_aug.preprocessing import ClassesNum, UsersNum
from getFisherDiagonal import getFisherDiagonal_initial, load_fisher_matrix
from simclr import SimCLR, MyNet, LIMU_encoder
from utils import MoCo_evaluate, evaluate, identify_users_number, load_model_config, CPC_evaluate
from torchvision.transforms import transforms
from data_aug import imu_transforms

parser = argparse.ArgumentParser(description='PyTorch SimCLR for Wearable Sensing')


parser.add_argument('-ft', '--if-fine-tune', default=True, type=bool, help='to decide whether tune all the layers')
parser.add_argument('-percent', default=1, type=float, help='how much percent of labels to use')
parser.add_argument('-shot', default=10, type=int, help='how many shots of labels to use')

parser.add_argument('--pretrained', default='CDL_slr0.7_v0/HHAR', type=str,
                    help='path to ContrastSense pretrained checkpoint')
parser.add_argument('-name', default='HHAR',
                    help='datasets name', choices=['HHAR', 'MotionSense', 'UCI', 'Shoaib', 'ICHAR', 'HASC'])
parser.add_argument('--store', default='test', type=str, help='define the name head for model storing')

parser.add_argument('-e', '--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--log-every-n-steps', default=4, type=int,
                    help='Log every n steps')
parser.add_argument('-t', '--temperature', default=1, type=float,
                    help='softmax temperature (default: 1)')

parser.add_argument('-g', '--gpu-index', default=1, type=int, help='Gpu index.')
parser.add_argument('--evaluate', default=False, type=bool, help='To decide whether to evaluate')
parser.add_argument('--resume', default='', type=str, help='To restart the model from a previous model')

parser.add_argument('--mol', default='MoCo', type=str, help='which model to use', choices=['SimCLR', 'LIMU', 'CPC', 'MoCo', 'DeepSense'])

parser.add_argument('-d', default=32, type=int, help='how many dims for encoder')

parser.add_argument('-cd', '--classifer_dims', default=1024, type=int, help='the feature dims of the classifier')
parser.add_argument('-final_dim', default=8, type=int, help='the output dims of the GRU')
parser.add_argument('-mo', default=0.9, type=float, help='the momentum for Batch Normalization')

parser.add_argument('-drop', default=0.1, type=float, help='the dropout portion')
parser.add_argument('-version', default="shot2", type=str, help='control the version of the setting')
parser.add_argument('-DAL', default=False, type=bool, help='Use Domain Adaversarial Learning or not')
parser.add_argument('-ad-lr', default=0.001, type=float, help='DAL learning rate')
parser.add_argument('-slr', default=0.5, type=float, help='DAL learning ratio')
parser.add_argument('-ewc', default=True, type=bool, help='Use EWC or not')
parser.add_argument('-ewc_lambda', default=5, type=float, help='EWC para')
parser.add_argument('-fishermax', default=0.01, type=float, help='fishermax')
parser.add_argument('-cl_slr', default=[0.7], nargs='+', type=float, help='the ratio of sup_loss')
parser.add_argument('-moco_K', default=1024, type=int, help='keys size')
parser.add_argument('-aug', default=False, type=bool, help='decide use data augmentation or not')
parser.add_argument('-mixup', default=False, type=bool, help='decide use mixup or not')
parser.add_argument('-p', default=0.2, type=float, help='possibility for one aug')
parser.add_argument('-cross', default='users', type=str, help='decide to use which kind of labels')

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def main(args, fisher=None):
    args.transfer = True
    if args.store == None:
        args.store = deepcopy(args.pretrained)
        
    if args.ewc:
        fisher_new = load_fisher_matrix(args.pretrained, args.device)
    
    dif = 0
    for n in fisher.keys():
        dif += np.linalg.norm(fisher_new[n].cpu() - fisher[n].cpu())
    
    print(dif)

if __name__ == '__main__':
    seed_torch()
    args = parser.parse_args()
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_index}')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    if args.ewc:
        fisher = getFisherDiagonal_initial(args)
    else:
        fisher = None
    main(args, fisher)

    # main(args)