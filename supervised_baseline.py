import argparse
import os

import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from simclr import SimCLR, MyNet, BaseLine, SeqActivityClassifier
from utils import evaluate

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR for Wearable Sensing')

parser.add_argument('-if-fine-tune', default=False, type=bool, help='to decide whether tune all the layers')
parser.add_argument('-if-val', default=False, type=bool, help='to decide whether use validation set')
parser.add_argument('-if-semi', default=False, type=bool, help='use 0.8*0.25 dataset to train the baseline')
parser.add_argument('-percent', default=1, type=int, help='how much percent of labels to use')
parser.add_argument('-lstm', default=False, type=bool,
                    help='decide to use which baseline')

parser.add_argument('-root', metavar='DIR', default='./datasets/processed',
                    choices=['./datasets/processed', './datasets/uot_processed', './datasets/HHAR'],
                    help='path to datasets')
parser.add_argument('-datasets-name', default='UciHarDataset',
                    help='datasets name', choices=['UciHarDataset', 'UotHarDataset'])

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=512, type=int,
                    help='feature dimension (default: 512)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.01, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--transfer', default=True, type=bool,
                    help='to tell whether we are doing transfer learning')
parser.add_argument('--evaluate', default=False, type=bool, help='To decide whether to evaluate')
parser.add_argument('--resume', default='', type=str, help='To restart the model from a previous model')
parser.add_argument('--best-acc', default=0., type=float, help='The initial best accuracy')
parser.add_argument('--store', default='lr', type=str, help='define the name head for model storing')


def main():
    args = parser.parse_args()
    print(args.if_val)
    print(args.if_semi)
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset(args.root, transfer=True)  # set transfer = True to avoid the contrastive setting

    if args.if_semi:
        train_dataset = dataset.get_dataset(args.datasets_name, args.n_views, 'tune', percent=args.percent)
    else:
        train_dataset = dataset.get_dataset(args.datasets_name, args.n_views, 'train')
    val_dataset = dataset.get_dataset(args.datasets_name, args.n_views, 'val')
    test_dataset = dataset.get_dataset(args.datasets_name, args.n_views, 'test')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    if args.lstm:
        model = SeqActivityClassifier(out_dim=args.out_dim,
                                      classes=6)  # set transfer to use the same head with transfer learning
    else:
        model = MyNet(transfer=True, out_dim=args.out_dim)    # set transfer to use the same head with transfer learning

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

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
        baseline = BaseLine(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        if args.evaluate:
            test_acc = evaluate(model=baseline.model, criterion=baseline.criterion, args=baseline.args, data_loader=test_loader)
            print('test acc: {}'.format('%.3f' % test_acc))
            return
        baseline.train(train_loader, val_loader)
        best_model_dir = os.path.join(baseline.writer.log_dir, 'model_best.pth.tar')
        baseline.test_performance(best_model_dir=best_model_dir, test_loader=test_loader)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
