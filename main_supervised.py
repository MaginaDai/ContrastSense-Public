import argparse
from copy import deepcopy
import os

import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
from torchvision import models

from MoCo import MoCo_model, MoCo
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, Dataset4Training
from data_aug.preprocessing import ClassesNum, UsersNum
from simclr import SimCLR, MyNet, LIMU_encoder
from utils import MoCo_evaluate
from torchvision.transforms import transforms
import torch.multiprocessing
from data_aug import imu_transforms

torch.multiprocessing.set_sharing_strategy('file_system')

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR for Wearable Sensing')


parser.add_argument('-ft', '--if-fine-tune', default=True, type=bool, help='to decide whether tune all the layers')
parser.add_argument('-shot', default=10, type=int, help='how many shots of labels to use')
parser.add_argument('-name', default='Shoaib', help='datasets name', choices=['HHAR', 'MotionSense', 'UCI', 'Shoaib', 'ICHAR', 'HASC'])
parser.add_argument('--store', default='test', type=str, help='define the name head for model storing')

parser.add_argument('-e', '--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=512, type=int,
                    help='feature dimension (default: 512)')
parser.add_argument('--log-every-n-steps', default=4, type=int, help='Log every n steps')
parser.add_argument('-t', '--temperature', default=1, type=float, help='softmax temperature (default: 1)')

parser.add_argument('-g', '--gpu-index', default=2, type=int, help='Gpu index.')
parser.add_argument('--evaluate', default=False, type=bool, help='To decide whether to evaluate')
parser.add_argument('--resume', default='', type=str, help='To restart the model from a previous model')

parser.add_argument('--mol', default='MoCo', type=str, help='which model to use', choices=['SimCLR', 'LIMU', 'CPC', 'MoCo', 'DeepSense'])

parser.add_argument('-d', default=32, type=int, help='how many dims for encoder')

parser.add_argument('-cd', '--classifer_dims', default=1024, type=int, help='the feature dims of the classifier')
parser.add_argument('-final_dim', default=8, type=int, help='the output dims of the GRU')
parser.add_argument('-mo', default=0.9, type=float, help='the momentum for Batch Normalization')

parser.add_argument('-drop', default=0.1, type=float, help='the dropout portion')
parser.add_argument('-version', default="supervised_plain", type=str, help='control the version of the setting')
parser.add_argument('-DAL', default=False, type=bool, help='Use Domain Adaversarial Learning or not')
parser.add_argument('-ad-lr', default=0.001, type=float, help='DAL learning rate')
parser.add_argument('-slr', default=0.5, type=float, help='DAL learning ratio')

parser.add_argument('--setting', default='sparse', type=str, choices=['full', 'sparse'], help='decide use tune or others')


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

def main():
    seed_torch()
    args = parser.parse_args()

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_index}')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    args.transfer = True
    if args.store == None:
        args.store = deepcopy(args.pretrained)

    dataset = ContrastiveLearningDataset(transfer=True, version=args.version, datasets_name=args.name)

    if args.setting == 'sparse':
        train_dataset = Dataset4Training(args.name, args.version, transform=transforms.Compose([imu_transforms.ToTensor()]), split='tune', transfer=True, shot=args.shot)
    elif args.setting == 'full':
        train_dataset = Dataset4Training(args.name, args.version, transform=transforms.Compose([imu_transforms.ToTensor()]), split='train', transfer=True)
    else:
        NotADirectoryError
    
    val_dataset = dataset.get_dataset('val')
    test_dataset = dataset.get_dataset('test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False, drop_last=False)
    
    model = MoCo_model(transfer=True, out_dim=args.out_dim, classes=ClassesNum[args.name], dims=args.d, 
                       classifier_dim=args.classifer_dims, final_dim=args.final_dim, momentum=args.mo, drop=args.drop, DAL=args.DAL, users_class=UsersNum[args.name])

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, last_epoch=-1)


    with torch.cuda.device(args.gpu_index):
        moco = MoCo(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        if args.evaluate:
            test_acc, test_f1 = MoCo_evaluate(model=moco.model, criterion=moco.criterion, args=moco.args, data_loader=test_loader)
            print('test f1: {}'.format('%.3f' % test_f1))
            print('test acc: {}'.format('%.3f' % test_acc))
            return

        moco.transfer_train(train_loader, val_loader)
        best_model_dir = os.path.join(moco.writer.log_dir, 'model_best.pth.tar')
        moco.test_performance(best_model_dir=best_model_dir, test_loader=test_loader)
    
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()