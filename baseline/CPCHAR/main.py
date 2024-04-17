import argparse
import sys
import torch
from os.path import dirname
sys.path.append(dirname(dirname(sys.path[0])))
sys.path.append(dirname(sys.path[0]))
from baseline.CPCHAR.CPC import CPC, CPC_model
from baseline.CPCHAR.dataload import CPCHAR_Dataset
from utils import seed_torch

parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning for Wearable Sensing')

#----------- paper related -----------------
parser.add_argument('--num_steps_prediction', type=int, default=28, help='Number of steps in the future to predict')
parser.add_argument('--kernel_size', type=int, default=3, help='Size of the conv filters in the encoder')
parser.add_argument('-lr', '--learning-rate', default=5e-4, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
parser.add_argument('-e', '--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-name', default='HHAR', help='datasets name', choices=['HHAR', 'MotionSense', 'UCI', 'Shoaib', 'HASC', 'ICHAR', "Merged_dataset"])

parser.add_argument('--resume', default='', type=str, help='To restart the model from a previous model')
parser.add_argument('--transfer', default=False, type=str, help='to tell whether we are doing transfer learning')
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')
parser.add_argument('--log-every-n-steps', default=5, type=int, help='Log every n steps')
parser.add_argument('-g', '--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 5)')
parser.add_argument('--store', default='lr', type=str, help='define the name head for model storing')
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
    
    args.padding = int(args.kernel_size // 2)
    args.input_size = 6
    
    dataset = CPCHAR_Dataset(transfer=False, version=args.version, datasets_name=args.name)

    train_dataset = dataset.get_dataset(split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=False, drop_last=True)
    
    val_dataset = dataset.get_dataset(split='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=False, drop_last=True)
    

    model = CPC_model(args)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.8)
    
    with torch.cuda.device(args.gpu_index):
        cpc = CPC(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        cpc.train(train_loader, val_loader)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    