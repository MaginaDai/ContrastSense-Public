import argparse
from copy import deepcopy
import sys
import os
import torch
from os.path import dirname


sys.path.append(dirname(dirname(sys.path[0])))
sys.path.append(dirname(sys.path[0]))

from data_aug.preprocessing import ClassesNum, UsersNum
from baseline.MMD.FMUDA import FMUDA, FM_model
from baseline.MMD.dataload import FMUDA_Dataset
from baseline.CPCHAR.dataload import CPCHAR_Dataset
from utils import evaluate, seed_torch

parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning for Wearable Sensing')

parser.add_argument('-version', default="shot", type=str, help='control the version of the setting')
parser.add_argument('-name', default='HHAR', help='datasets name', choices=['HHAR', 'MotionSense', 'Shoaib', 'HASC'])
parser.add_argument('--store', default='FMUDA_HHAR', type=str, help='define the name head for model storing')
parser.add_argument('-m', '--method', default='CM', type=str, help='select method to use', choices=['FM', 'CM'])

parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('-b', '--batch-size', default=125, type=int, metavar='N')
parser.add_argument('-g', '--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')

parser.add_argument('-e', '--epochs', default=2000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-shot', default=10, type=int, help='how many shots of labels to use')
parser.add_argument('-percent', default=1, type=float, help='how much percent of labels to use')
parser.add_argument('-j', '--workers', default=5, type=int, metavar='N', help='number of data loading workers (default: 5)')
parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')



def main():
    args = parser.parse_args()

    seed_torch(seed=args.seed)

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1


    dataset = FMUDA_Dataset(transfer=False, version=args.version, datasets_name=args.name)
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

    user_num = UsersNum[args.name]
    model = FM_model(classes=ClassesNum[args.name], method=args.method, domains=user_num)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    with torch.cuda.device(args.gpu_index):
        FM = FMUDA(model=model, optimizer=optimizer, args=args)
        FM.train(train_loader, tune_loader, val_loader, test_loader)
        best_model_dir = os.path.join(FM.writer.log_dir, 'model_best.pth.tar')
        FM.test_performance(best_model_dir=best_model_dir, test_loader=test_loader)

if __name__ == '__main__':
    main()