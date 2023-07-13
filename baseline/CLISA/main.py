import argparse
import sys
from os.path import dirname
sys.path.append(dirname(dirname(sys.path[0])))
sys.path.append(dirname(sys.path[0]))
import torch
from dataloader import load_CLISA_data

from utils import seed_torch

parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning for Wearable Sensing')


parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--transfer', default=False, type=str, help='to tell whether we are doing transfer learning')
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')
parser.add_argument('--resume', default='', type=str, help='To restart the model from a previous model')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('-b', '--batch-size', default=512, type=int)

parser.add_argument('-name', default='SEED', help='datasets name', choices=['SEED', 'SEED_IV'])
parser.add_argument('--log-every-n-steps', default=5, type=int, help='Log every n steps')
parser.add_argument('-g', '--gpu-index', default=2, type=int, help='Gpu index.')
parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('-t', '--temperature', default=0.1, type=float, help='softmax temperature (default: 0.1)')

parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 5)')
parser.add_argument('--store', default='lr', type=str, help='define the name head for model storing')
parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-version', default="shot", type=str, help='control the version of the setting')

def main():
    args = parser.parse_args()
    seed_torch(seed=args.seed)
    
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_index}')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    
    train_loader = load_CLISA_data(transfer=False, version=args.version, datasets_name=args.name)

    model = CLISA_model('CL')
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, last_epoch=-1)
    
    with torch.cuda.device(args.gpu_index):
        clisa = CLISA(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        clisa.train(train_loader)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    