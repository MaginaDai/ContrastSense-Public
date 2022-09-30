import argparse
import os

import torch
import torch.backends.cudnn as cudnn
from torchvision import models

from MinMax.MinMaxDataset import MinMaxDataset
from MinMax.Selector import DeepConvLSTM, MoCoMinMax
from MoCo import MoCo_v1, MoCo
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from simclr import SimCLR, MyNet, LIMU_encoder
from utils import load_model_config

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning for Wearable Sensing')


parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-lr-selector', '--learning-rate-selector', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr_selector')
parser.add_argument('--transfer', default=False, type=str,
                    help='to tell whether we are doing transfer learning')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--resume', default='', type=str, help='To restart the model from a previous model')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')

parser.add_argument('--out_dim', default=512, type=int,
                    help='feature dimension (default: 512)')
parser.add_argument('-t', '--temperature', default=1, type=float,
                    help='softmax temperature (default: 1)')
parser.add_argument('--store', default='lr', type=str, help='define the name head for model storing')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-root', metavar='DIR', default='../datasets/HHAR',
                    choices=['../datasets/processed', '../datasets/uot_processed', '../datasets/HHAR', '../datasets/HHAR filter'],
                    help='path to datasets')
parser.add_argument('-datasets-name', default='HHAR',
                    help='datasets name', choices=['UciHarDataset', 'UotHarDataset', 'HHAR'])
parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('-g', '--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--best-acc', default=0., type=float, help='The initial best accuracy')
parser.add_argument('-mol', default='MoCo', type=str, help='which model to use', choices=['SimCLR', 'LIMU', 'CPC', 'MoCo'])
parser.add_argument('-d', default=256, type=int, help='how dim for CPC')
parser.add_argument('-moco_K', default=4096, type=int, help='keys size')
parser.add_argument('-moco_m', default=0.999, type=int, help='momentum value')


def main():
    args = parser.parse_args()
    # check if gpu training is available

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = MinMaxDataset(args.root, transfer=False)

    train_dataset = dataset.get_dataset(args.datasets_name, 'train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    model = MoCo_v1(transfer=False, out_dim=args.out_dim, K=args.moco_K, m=args.moco_m, T=args.temperature, classes=6, dims=args.d)
    selector = DeepConvLSTM()

    optimizer1 = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    optimizer2 = torch.optim.Adam(selector.parameters(), args.lr_selector, weight_decay=args.weight_decay)

    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=len(train_loader), eta_min=0, last_epoch=-1)

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
            optimizer1.load_state_dict(checkpoint['optimizer1'])
            optimizer2.load_state_dict(checkpoint['optimizer2'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        moco_minmax = MoCoMinMax(model=model, selector=selector, optimizer1=optimizer1, scheduler1=scheduler1,
                                 optimizer2=optimizer2, scheduler2=scheduler2, args=args)
        moco_minmax.train(train_loader)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

