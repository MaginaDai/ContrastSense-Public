import argparse
import torch, os, random
import numpy as np
import matplotlib.pyplot as plt
from getFisherDiagonal import load_fisher_matrix, getFisherDiagonal_initial

parser = argparse.ArgumentParser(description='PyTorch SimCLR for Wearable Sensing')

parser.add_argument('-name', default='HHAR', help='datasets name', choices=['HHAR', 'MotionSense', 'UCI', 'Shoaib', 'ICHAR', 'HASC'])
parser.add_argument('-version', default="shot2", type=str, help='control the version of the setting')
parser.add_argument('--mol', default='MoCo', type=str, help='which model to use', choices=['SimCLR', 'LIMU', 'CPC', 'MoCo', 'DeepSense'])
parser.add_argument('-moco_K', default=1024, type=int, help='keys size')
parser.add_argument('-pretrain_lr', default=1e-4, type=float, help='learning rate during pretraining')
parser.add_argument('-cross', default='users', type=str, help='decide to use which kind of labels')
parser.add_argument('-label_type', default=1, type=int, help='How many different kinds of labels for pretraining')
parser.add_argument('-g', '--gpu-index', default=1, type=int, help='Gpu index.')
parser.add_argument('-fishermax', default=0.01, type=float, help='fishermax')
parser.add_argument('-cl_slr', default=[0.7], nargs='+', type=float, help='the ratio of sup_loss')

parser.add_argument('--pretrained', default='CDL_slr0.7_v0/HHAR', type=str, help='path to ContrastSense pretrained checkpoint')

parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')


def seed_torch(seed=3):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return


def ewc_analysis(args):
    seed_torch(seed=args.seed)

    args.device = torch.device(f'cuda:{args.gpu_index}')
    fisher = load_fisher_matrix(args.pretrained, args.device)
    fisher_new = getFisherDiagonal_initial(args)

    fisher_array = dict2array(fisher)
    fisher_new_array = dict2array(fisher_new)

    cdf_plot(fisher_array, fisher_new_array)
    return

def ewc_cmp_with_different_seed(args):
    args = parser.parse_args()

    args.device = torch.device(f'cuda:{args.gpu_index}')
    seed_torch(seed=0)
    fisher_1 = getFisherDiagonal_initial(args)

    seed_torch(seed=100)
    fisher_2 = getFisherDiagonal_initial(args)


    fisher_1 = dict2array(fisher_1)
    fisher_2 = dict2array(fisher_2)

    cdf_plot(fisher_1, fisher_2)
    return


def dict2array(fisher):
    para = np.empty(shape=0)
    dict_name = list(fisher)
    for name in dict_name:
        para = np.append(para, fisher[name].cpu().numpy().reshape(-1)[:])
    return para


def cdf_plot(fisher, fisher_new):
    x_fisher = np.sort(fisher)
    x_fisher_new = np.sort(fisher_new)
    y = 1. * np.arange(len(fisher)) / (len(fisher) - 1)

    plt.figure()
    plt.plot(x_fisher, y)
    plt.plot(x_fisher_new, y)
    plt.xlim([1e-7, 1e-4])
    plt.legend(['seed 1', 'seed 2'])
    plt.savefig('fisher_cdf_seed_cmp.png')
    return


if __name__ == '__main__':
    args = parser.parse_args()
    # ewc_analysis(args)

    ewc_cmp_with_different_seed(args)