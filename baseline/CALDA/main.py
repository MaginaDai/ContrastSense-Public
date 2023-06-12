import argparse
import sys
import os
from os.path import dirname
sys.path.append(dirname(dirname(sys.path[0])))
sys.path.append(dirname(sys.path[0]))

from baseline.Mixup.myo_dataload import Myo_Dataset
from baseline.CALDA.model import CALDA_encoder, CALDA
from baseline.CPCHAR.dataload import CPCHAR_Dataset
from data_aug.preprocessing import ClassesNum, UsersNum

import torch
from utils import seed_torch

parser = argparse.ArgumentParser(description='Baseline CALDA')

parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--transfer', default=False, type=str, help='to tell whether we are doing transfer learning')
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')
parser.add_argument('--resume', default='', type=str, help='To restart the model from a previous model')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('-b', '--batch-size', default=128, type=int, help="batch size")
parser.add_argument('-name', default='Myo', help='datasets name', choices=['HHAR', 'MotionSense', 'Shoaib', 'HASC', 'Myo', 'NinaPro'])
parser.add_argument('--log-every-n-steps', default=5, type=int, help='Log every n steps')
parser.add_argument('-g', '--gpu-index', default=2, type=int, help='Gpu index.')
parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 5)')
parser.add_argument('--store', default='lr', type=str, help='define the name head for model storing')
parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-version', default="shot0", type=str, help='control the version of the setting')
parser.add_argument('-percent', default=1, type=float, help='how much percent of labels to use')
parser.add_argument('-shot', default=10, type=int, help='how many shots of labels to use')

def main():
    args = parser.parse_args()
    args.num_units = 128
    args.weight_adversary = 1.0
    args.weight_similarity = 10.0
    args.temperature = 0.1

    seed_torch(seed=args.seed)
    
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_index}')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    if args.name == 'Myo' or args.name == 'NinaPro':
        args.modal = 'emg'
        dataset = Myo_Dataset(transfer=True, version=args.version, datasets_name=args.name)
    else:
        args.modal = 'imu'
        dataset = CPCHAR_Dataset(transfer=True, version=args.version, datasets_name=args.name)
    
    tune_dataset = dataset.get_dataset('tune', percent=args.percent, shot=args.shot)
    val_dataset = dataset.get_dataset('val')
    test_dataset = dataset.get_dataset('test')

    train_loader = torch.utils.data.DataLoader(
        tune_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=False)
    
    model = CALDA_encoder(num_classes=ClassesNum[args.name], 
                          num_domains=UsersNum[args.name],
                          num_units=args.num_units,
                          modal=args.modal)
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    with torch.cuda.device(args.gpu_index):
        calda = CALDA(model=model, optimizer=optimizer, args=args)
        calda.train(train_loader, val_loader)

        best_model_dir = os.path.join(calda.writer.log_dir, 'model_best.pth.tar')
        calda.test_performance(best_model_dir=best_model_dir, test_loader=test_loader)

if __name__ == '__main__':
    main()