import argparse
import os
import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

from ContrastSense import ContrastSense, ContrastSense_v1
from data_loader.contrastive_learning_dataset import ContrastiveLearningDataset
from getPenalty import getPenalty
from utils import seed_torch



# https://arxiv.org/abs/1911.05722

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning for Wearable Sensing')

    parser.add_argument('-name', default='Myo', help='datasets name', choices=['HHAR', 'MotionSense', 'Shoaib', 'HASC', 'Myo', 'NinaPro', "Merged_dataset"])
    parser.add_argument('-version', default="shot0", type=str, help='control the version of the setting')

    parser.add_argument('-cross', default='users', type=str, help='decide to use which kind of labels')
    parser.add_argument('--store', default='test', type=str, help='define the name head for model storing')

    parser.add_argument('--transfer', default=False, type=str, help='to tell whether we are doing transfer learning')
    parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')
    parser.add_argument('--resume', default='', type=str, help='To restart the model from a previous model')


    parser.add_argument('-g', '--gpu-index', default=0, type=int, help='Gpu index.')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 10)')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')
    parser.add_argument('--log-every-n-steps', default=5, type=int, help='Log every n steps')
    parser.add_argument('--best-acc', default=0., type=float, help='The initial best accuracy')

    parser.add_argument('-mol', default='ContrastSense', type=str, help='which model to use', choices=['SimCLR', 'LIMU', 'CPC', 'ContrastSense'])
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-t', '--temperature', default=0.1, type=float, help='softmax temperature (default: 0.1)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,help='mini-batch size (default: 256),')
    parser.add_argument('-e', '--epochs', default=2000, type=int, metavar='N',help='number of total epochs to run')
    parser.add_argument('-ContrastSense_K', default=1024, type=int, help='keys size')
    parser.add_argument('-ContrastSense_m', default=0.999, type=float, help='momentum value')

    parser.add_argument('-label_type', default=1, type=int, help='How many different kinds of labels for pretraining')
    parser.add_argument('-slr', default=[0.7], nargs='+', type=float, help='the ratio of sup_loss')
    parser.add_argument('-tem_labels', default=[0.1], nargs='+', type=float, help='the temperature for supervised CL')

    parser.add_argument('-penalty', default=True, type=float, help='Use parameter-wise-penalty or not')
    parser.add_argument('-fishermax', default=100, type=float, help='fishermax')
  
    parser.add_argument('-hard', default=True, type=bool, help='hard sampling or not')  # we sample hard ones from the data.
    parser.add_argument('-hard_record', default=False, type=bool, help='record hardest samples related information or not')  # we sample hard ones from the data.
    parser.add_argument('-sample_ratio', default=0.05, type=float, help='hard sampling or not')  # we eliminate hardest ones from the data.
    parser.add_argument('-last_ratio', default=1.0, type=float, help='ratio of hard sample to preserve')  # we sample hard ones from the other domains.
    parser.add_argument('-scale_ratio', default=1.0, type=float, help='to scale the similarity between domains')  # to scale the similarity between domains
    parser.add_argument('-time_window', default=0, type=float, help='[time_label-t, time_label + t]')  # how much time idx labels are included.

    args = parser.parse_args()
    seed_torch(seed=args.seed)

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_index}')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    
    if args.name in ['NinaPro', 'Myo', 'UCI']:
        args.modal = 'emg'
    else:
        args.modal = 'imu'
    
    if args.last_ratio != 1.0:
        if args.name == 'HHAR':
            args.last_ratio = 0.5
        elif args.name == 'MotionSense' or args.name == 'Shoaib':
            args.last_ratio = 0.8
        elif args.name == 'HASC':
            args.last_ratio = 0.7
        else:
            pass  # the other datasets just use the input parameter. 
    
    if args.label_type == 1:
        if args.name == 'NinaPro':
            args.slr=[0.1]
        if args.name == 'HASC':
            args.tem_labels = [0.08,]

    return args


def main():
    args = get_args()

    dataset = ContrastiveLearningDataset(transfer=False, version=args.version, datasets_name=args.name, modal=args.modal)

    train_dataset = dataset.get_dataset(split='train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    model = ContrastSense_v1(args)
   
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, last_epoch=-1)

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
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    with torch.cuda.device(args.gpu_index):
        contrast = ContrastSense(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        contrast.train(train_loader)

    if args.penalty and args.label_type != 0:
        getPenalty(args, train_loader, contrast.writer.log_dir)
    return

if __name__ == '__main__':
    main()