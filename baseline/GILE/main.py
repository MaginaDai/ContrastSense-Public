# encoding=utf-8
from copy import deepcopy
import sys
import os
import torch
from os.path import dirname

sys.path.append(dirname(dirname(sys.path[0])))
sys.path.append(dirname(sys.path[0]))

from baseline.CPCHAR.dataload import CPCHAR_Dataset
from baseline.GILE.dataloader import load_GILE_type_data
from baseline.GILE.GILE import train
import baseline.GILE.network as net
from data_aug.preprocessing import ClassesNum, UsersNum, PositionNum, DevicesNum

from utils import seed_torch
import torch
import argparse

parser = argparse.ArgumentParser(description='argument setting of network')

parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('-version', default="shot", type=str, help='control the version of the setting')
parser.add_argument('--store', default='lr', type=str, help='define the name head for model storing')
parser.add_argument('-name', default='HHAR', help='datasets name', choices=['HHAR', 'MotionSense', 'UCI', 'Shoaib', 'HASC', 'ICHAR'])
parser.add_argument('-percent', default=1, type=float, help='how much percent of labels to use')
parser.add_argument('-shot', default=10, type=int, help='how many shots of labels to use')

parser.add_argument('--now_model_name', type=str, default='GILE', help='the type of model')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=150, help='number of training epochs')
parser.add_argument('-g', '--gpu-index', default=0, type=int, help='Gpu index.')

parser.add_argument('--n_feature', type=int, default=6, help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=128, help='length of sliding window')
parser.add_argument('--d_AE', type=int, default=50, help='dim of AE')
parser.add_argument('--sigma', type=float, default=1, help='parameter of mmd')
parser.add_argument('--weight_mmd', type=float, default=1.0, help='weight of mmd loss')

parser.add_argument('--test_every', type=int, default=1, help='do testing every n epochs')
parser.add_argument('-n_target_domains', type=int, default=1, help='number of target domains')

parser.add_argument('--beta', type=float, default=1., help='multiplier for KL')

parser.add_argument('--x-dim', type=int, default=1152, help='input size after flattening')
parser.add_argument('--aux_loss_multiplier_y', type=float, default=1000., help='multiplier for y classifier')
parser.add_argument('--aux_loss_multiplier_d', type=float, default=1000., help='multiplier for d classifier')

parser.add_argument('--beta_d', type=float, default=1., help='multiplier for KL d')
parser.add_argument('--beta_x', type=float, default=0., help='multiplier for KL x')
parser.add_argument('--beta_y', type=float, default=1., help='multiplier for KL y')

parser.add_argument('--weight_true', type=float, default=1000.0, help='weights for classifier true')
parser.add_argument('--weight_false', type=float, default=1000.0, help='weights for classifier false')
parser.add_argument('--setting', default='sparse', type=str, choices=['full', 'sparse'], help='decide use tune or others')
parser.add_argument('-cross', default='positions', type=str, help='decide to use which kind of labels')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.cross == 'users':
        args.n_domains = UsersNum[args.name]
    elif args.cross == 'positions':
        args.n_domains = PositionNum[args.name]
    elif args.cross == 'devices':
        args.n_domains = DevicesNum[args.name]

    args.n_class = ClassesNum[args.name]
    seed_torch(seed=args.seed)
    DEVICE = torch.device(f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')
    args.device = DEVICE
    tune_loader, val_loader, test_loader = load_GILE_type_data(args.name, args.version, args.shot, args.batch_size, args.setting, args.cross)
    model = net.load_model(args)
    model = model.to(DEVICE)
    optimizer = net.set_up_optimizers(model.parameters())
    with torch.cuda.device(args.gpu_index):
        train(model, DEVICE, optimizer, tune_loader, val_loader, test_loader, args)