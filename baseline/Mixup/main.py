import argparse
from copy import deepcopy
import sys
import os
import torch
from os.path import dirname

sys.path.append(dirname(dirname(sys.path[0])))
sys.path.append(dirname(sys.path[0]))

from baseline.ClusterHAR.TPN_model import Transfer_Coder
from data_aug.preprocessing import ClassesNum, UsersNum
from baseline.CPCHAR.dataload import CPCHAR_Dataset
from baseline.Mixup.MIX import MIX
from utils import evaluate, seed_torch
from baseline.Mixup.myo_dataload import Myo_Dataset
from baseline.Mixup.ConvNet import ConvNet


parser = argparse.ArgumentParser(description='PyTorch Mixup for Wearable Sensing')

parser.add_argument('-version', default="leave_shot0", type=str, help='control the version of the setting')
parser.add_argument('-name', default='HASC', help='datasets name', choices=['HHAR', 'MotionSense', 'Shoaib', 'HASC', 'Myo', 'NinaPro'])
parser.add_argument('--store', default='test_Ninapro', type=str, help='define the name head for model storing')

parser.add_argument('-lr', '--learning-rate', default=5e-3, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('-b', '--batch-size', default=100, type=int, metavar='N')
parser.add_argument('-g', '--gpu-index', default=1, type=int, help='Gpu index.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
  
parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-shot', default=10, type=int, help='how many shots of labels to use')
parser.add_argument('-percent', default=1, type=float, help='how much percent of labels to use')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
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

    if args.name == 'Myo' or args.name == 'NinaPro':
        dataset = Myo_Dataset(transfer=True, version=args.version, datasets_name=args.name)
    else:
        dataset = CPCHAR_Dataset(transfer=True, version=args.version, datasets_name=args.name)
    
    tune_dataset = dataset.get_dataset('tune', percent=args.percent, shot=args.shot)
    val_dataset = dataset.get_dataset('val')
    test_dataset = dataset.get_dataset('test')

    tune_loader = torch.utils.data.DataLoader(
        tune_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=False)

    if args.name == 'Myo' or args.name == 'NinaPro':
        model = ConvNet(number_of_class=ClassesNum[args.name])
    else:
        model = Transfer_Coder(classes=ClassesNum[args.name], method='CL')
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    with torch.cuda.device(args.gpu_index):
        mixup = MIX(model=model, optimizer=optimizer, args=args)
        mixup.train(tune_loader, val_loader)
        best_model_dir = os.path.join(mixup.writer.log_dir, 'model_best.pth.tar')
        mixup.test_performance(best_model_dir=best_model_dir, test_loader=test_loader)

if __name__ == '__main__':
    main()