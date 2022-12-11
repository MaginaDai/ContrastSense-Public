# encoding=utf-8
import matplotlib

from baseline.CPCHAR.dataload import CPCHAR_Dataset
matplotlib.use('Agg')

from copy import deepcopy
import sys
import os
import torch
from os.path import dirname

sys.path.append(dirname(dirname(sys.path[0])))
sys.path.append(dirname(sys.path[0]))

from GILE import train
import network as net

from utils import seed_torch, set_name
import torch
import argparse

parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--now_model_name', type=str, default='GILE', help='the type of model')

parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=150, help='number of training epochs')
parser.add_argument('--dataset', type=str, default='ucihar', help='name of dataset')
parser.add_argument('-g', '--gpu-index', default=0, type=int, help='Gpu index.')

parser.add_argument('--n_feature', type=int, default=9, help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=128, help='length of sliding window')
parser.add_argument('--n_class', type=int, default=6, help='number of class')
parser.add_argument('--d_AE', type=int, default=50, help='dim of AE')
parser.add_argument('--sigma', type=float, default=1, help='parameter of mmd')
parser.add_argument('--weight_mmd', type=float, default=1.0, help='weight of mmd loss')

parser.add_argument('--target_domain', type=str, default='0', help='the target domain, [0 to 4]')
parser.add_argument('--test_every', type=int, default=1, help='do testing every n epochs')
parser.add_argument('-n_domains', type=int, default=5, help='number of total domains actually')
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

if __name__ == '__main__':
    args = parser.parse_args()
    
    seed_torch(seed=args.seed)

    DEVICE = torch.device(f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')

    args.device = DEVICE
    
    dataset = CPCHAR_Dataset(transfer=False, version=args.version, datasets_name=args.name)

    train_dataset = dataset.get_dataset(split='train')
    tune_dataset = dataset.get_dataset('tune', percent=args.percent, shot=args.shot)
    val_dataset = dataset.get_dataset('val')
    test_dataset = dataset.get_dataset('test')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    tune_loader = torch.utils.data.DataLoader(
        tune_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=False)

    
    model = net.load_model(args)
    model = model.to(DEVICE)
    optimizer = net.set_up_optimizers(model.parameters())
    train(model, DEVICE, optimizer, train_loader, tune_loader, val_loader, test_loader, args)