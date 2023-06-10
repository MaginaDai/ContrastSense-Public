import argparse
import torch, os, random
import numpy as np
import matplotlib.pyplot as plt
from getFisherDiagonal import load_fisher_matrix, getFisherDiagonal_initial

parser = argparse.ArgumentParser(description='PyTorch SimCLR for Wearable Sensing')

parser.add_argument('-name', default='Shoaib', help='datasets name', choices=['HHAR', 'MotionSense', 'Shoaib', 'HASC'])
parser.add_argument('-version', default="shot0", type=str, help='control the version of the setting')
parser.add_argument('--mol', default='MoCo', type=str, help='which model to use', choices=['SimCLR', 'LIMU', 'CPC', 'MoCo', 'DeepSense'])
parser.add_argument('-moco_K', default=1024, type=int, help='keys size')
parser.add_argument('-pretrain_lr', default=1e-4, type=float, help='learning rate during pretraining')
parser.add_argument('-cross', default='users', type=str, help='decide to use which kind of labels')
parser.add_argument('-label_type', default=1, type=int, help='How many different kinds of labels for pretraining')
parser.add_argument('-g', '--gpu-index', default=3, type=int, help='Gpu index.')
parser.add_argument('-fishermax', default=1e-4, type=float, help='fishermax')
parser.add_argument('-slr', default=[0.7], nargs='+', type=float, help='the ratio of sup_loss')

parser.add_argument('--pretrained', default='CDL_slr0.7_v0/Shoaib', type=str, help='path to ContrastSense pretrained checkpoint')

parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')

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
    fisher_new, _ = getFisherDiagonal_initial(args)

    cdf_plot(fisher, fisher_new)
    return

def ewc_cmp_with_different_seed(args):
    args = parser.parse_args()

    args.device = torch.device(f'cuda:{args.gpu_index}')
    seed_torch(seed=0)
    fisher_1, _ = getFisherDiagonal_initial(args)

    seed_torch(seed=1000)
    fisher_2, _ = getFisherDiagonal_initial(args)
    
    cdf_plot(fisher_1, fisher_2)
    return


def ewc_cmp_the_influcence_of_add_InfoNCE(args):
    args = parser.parse_args()

    args.device = torch.device(f'cuda:{args.gpu_index}')
    seed_torch(seed=0)
    fisher_cdl, fisher_infoNCE = getFisherDiagonal_initial(args)

    fisher = {n: torch.zeros(p.shape).to(args.device) for n, p in fisher_cdl.items()}
    fisher_ratio = {n: torch.zeros(p.shape).to(args.device) for n, p in fisher_cdl.items()}

    low_f = 10
    fmax = 0
    fmin = 10
    fmax_cdl = 0
    for n, p in fisher_infoNCE.items():
        if torch.min(fisher_infoNCE[n]) < fmin:
            fmin = torch.min(fisher_infoNCE[n])
        if torch.max(fisher_infoNCE[n]) > fmax:
            fmax = torch.max(fisher_infoNCE[n])
        
        if torch.max(fisher_cdl[n]) > fmax_cdl:
            fmax_cdl = torch.max(fisher_cdl[n])
    
    for n, p in fisher_cdl.items():
        fisher_ratio[n] = (fmax - fisher_infoNCE[n]) / (fmax - fmin) * fmax_cdl # make them around the same scale
        fisher[n] = 0.5 * fisher_ratio[n]  + 0.5 * fisher_cdl[n]
        
        if torch.min(fisher[n]) < low_f:
            low_f = torch.min(fisher[n])

    print(low_f)
    cdf_plot_cmp(fisher_cdl, fisher_ratio, fisher)
    return
    

def dict2array(fisher):
    para = np.empty(shape=0)
    dict_name = list(fisher)
    for name in dict_name:
        para = np.append(para, fisher[name].cpu().numpy().reshape(-1)[:])
    return para

def cdf_plot_cmp(x1, x2, x3):
    x1 = dict2array(x1)
    x2 = dict2array(x2)
    x3 = dict2array(x3)
    x1_s = np.sort(x1)
    x2_s = np.sort(x2)
    x3_s = np.sort(x3)
    y = 1. * np.arange(len(x1)) / (len(x1) - 1)

    plt.figure()
    plt.plot(x1_s, y)
    plt.plot(x2_s, y)
    plt.plot(x3_s, y)
    plt.legend(['cdl', 'infoNCE', 'combined'])
    plt.savefig('before and after combined.png')

    return


def cdf_plot(fisher, fisher_new):
    fisher = dict2array(fisher)
    fisher_new = dict2array(fisher_new)

    x_fisher = np.sort(fisher)
    x_fisher_new = np.sort(fisher_new)
    y = 1. * np.arange(len(fisher)) / (len(fisher) - 1)

    plt.figure()
    plt.plot(x_fisher, y)
    plt.plot(x_fisher_new, y)
    # plt.xlim([0.004, 0.006])
    plt.legend(['old', 'new'])
    plt.savefig('improved ewc.png')
    return


if __name__ == '__main__':
    args = parser.parse_args()
    # ewc_analysis(args)

    # ewc_cmp_with_different_seed(args)

    ewc_cmp_the_influcence_of_add_InfoNCE(args)