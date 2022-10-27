import argparse
from copy import deepcopy
import logging
import os
import pdb
import string

import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np

from torchvision import models

from CPC import CPCV1, CPC
from DeepSense import DeepSense_encoder
from MoCo import MoCo_model, MoCo_v1, MoCo_encoder, MoCo
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from data_aug.preprocessing import ClassesNum, UsersNum
from simclr import SimCLR, MyNet, LIMU_encoder
from utils import MoCo_evaluate, evaluate, identify_users_number, load_model_config, CPC_evaluate, seed_torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR for Wearable Sensing')

parser.add_argument('--pretrained', default='Origin_wo_HASC', type=str,
                    help='path to SimClR pretrained checkpoint')
parser.add_argument('-ft', '--if-fine-tune', default=True, type=bool, help='to decide whether tune all the layers')
parser.add_argument('-if-val', default=True, type=bool, help='to decide whether use validation set')
parser.add_argument('-percent', default=1, type=float, help='how much percent of labels to use')
parser.add_argument('-shot', default=10, type=int, help='how many shots of labels to use')

parser.add_argument('-name', default='Shoaib',
                    help='datasets name', choices=['HHAR', 'MotionSense', 'UCI', 'Shoaib', 'ICHAR', 'HASC'])

parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-e', '--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=512, type=int,
                    help='feature dimension (default: 512)')
parser.add_argument('--log-every-n-steps', default=4, type=int,
                    help='Log every n steps')
parser.add_argument('-t', '--temperature', default=1, type=float,
                    help='softmax temperature (default: 1)')
parser.add_argument('-g', '--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--evaluate', default=False, type=bool, help='To decide whether to evaluate')
parser.add_argument('--resume', default='', type=str, help='To restart the model from a previous model')
parser.add_argument('--store', default=None, type=str, help='define the name head for model storing')
parser.add_argument('--mol', default='MoCo', type=str, help='which model to use', choices=['SimCLR', 'LIMU', 'CPC', 'MoCo', 'DeepSense'])

parser.add_argument('--timestep', default=15, type=int, help='how many time steps for CPC')
parser.add_argument('-d', default=32, type=int, help='how dim for CPC')
parser.add_argument('-moco_K', default=4096, type=int, help='keys size')
parser.add_argument('-moco_m', default=0.999, type=float, help='momentum value')
parser.add_argument('-s_gamma', default=0.5, type=float, help='the gamma of StepLR')

parser.add_argument('-cd', '--classifer_dims', default=1024, type=int, help='the feature dims of the classifier')
parser.add_argument('-final_dim', default=8, type=int, help='the output dims of the GRU')
parser.add_argument('-mo', default=0.9, type=float, help='the momentum for Batch Normalization')

parser.add_argument('-drop', default=0.1, type=float, help='the dropout portion')
parser.add_argument('-version', default="shot", type=str, help='control the version of the setting')
parser.add_argument('-DAL', default=False, type=bool, help='Use Domain Adaversarial Learning or not')
parser.add_argument('-ad-lr', default=0.001, type=float, help='DAL learning rate')


def main():
    args = parser.parse_args()

    seed_torch(seed=args.seed)

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    args.transfer = True
    if args.store == None:
        args.store = deepcopy(args.pretrained)
    args.pretrained = './runs/' + args.pretrained + '/model_best.pth.tar'

    dataset = ContrastiveLearningDataset(transfer=True, version=args.version, datasets_name=args.name)
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


    if args.DAL:
        user_num = UsersNum[args.name]
        train_dataset =  dataset.get_dataset('train')
        train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=False)
    else:
        user_num = None
    
    if args.mol == 'LIMU':
        model_cfg = load_model_config(target='pretrain_base', prefix='base', version='v1')
        model = LIMU_encoder(model_cfg, transfer=True, out_dim=args.out_dim, classes=ClassesNum[args.name])
    elif args.mol == 'CPC':
        model = CPCV1(timestep=args.timestep, batch_size=args.batch_size, seq_len=104, transfer=True, classes=ClassesNum[args.name], dims=args.d)
    elif args.mol == 'MoCo':
        model = MoCo_model(transfer=True, out_dim=args.out_dim, classes=ClassesNum[args.name], dims=args.d, 
                           classifier_dim=args.classifer_dims, final_dim=args.final_dim, momentum=args.mo, drop=args.drop, DAL=args.DAL, users_class=user_num)
    elif args.mol == 'DeepSense':
        model = DeepSense_encoder(transfer=True, out_dim=args.out_dim, classes=ClassesNum[args.name], dims=args.d)
    else:
        model = MyNet(transfer=True, out_dim=args.out_dim, classes=ClassesNum[args.name], if_bn=args.if_bn, if_g=args.if_g, if_lstm=args.if_lstm)

    # model.to(args.device)

    classifier_name = []
    # load pre-trained model
    for name, param in model.named_parameters():
        if "classifier" in name or 'discriminator' in name:
            classifier_name.append(name)
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint['state_dict']

            # rename moco pre-trained keys
            if args.mol == 'MoCo':
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('encoder_q') and not k.startswith('encoder_q.linear2'):
                        # remove prefix
                        state_dict[k[len("encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

            args.start_epoch = 0
            log = model.load_state_dict(state_dict, strict=False)
            if not args.evaluate:
                if args.mol == 'CPC' or args.mol == 'MoCo':
                    assert log.missing_keys == classifier_name
                else:
                    assert log.missing_keys == ['linear2.weight', 'linear2.bias']
                    # init the fc layer
                    model.linear2.weight.data.normal_(mean=0.0, std=0.01)
                    model.linear2.bias.data.zero_()
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if not args.if_fine_tune:
        # froze the parameter
        for name, param in model.named_parameters():
            if args.mol == 'CPC' or args.mol == 'MoCo':
                if name not in classifier_name:
                    param.requires_grad = False
            else:
                if name not in ['classifier']:
                    param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, last_epoch=-1)
    if args.DAL:
        optimizer_DAL = torch.optim.Adam(model.parameters(), args.ad_lr, weight_decay=args.weight_decay)
        scheduler_DAL = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_DAL, T_max=args.epochs, last_epoch=-1)
        

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150], gamma=args.s_gamma, last_epoch=-1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            args.start_epoch = checkpoint['epoch'] + 1
            try:
                best_acc = checkpoint['best_acc']
            except AttributeError:
                best_acc = checkpoint['acc']
            best_acc = best_acc.to(args.device)  # best_acc1 may be from a checkpoint from a different GPU
            args.best_acc = best_acc
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if not args.if_fine_tune:
        # optimize only the linear classifier
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        if args.mol == "CPC" or args.mol == 'MoCo':
            assert len(parameters) == len(classifier_name)
        else:
            assert len(parameters) == 2 

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        if args.mol == 'CPC':
            cpc = CPC(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
            if args.evaluate:
                test_acc = CPC_evaluate(model=cpc.model, criterion=cpc.criterion, args=cpc.args, data_loader=test_loader)
                print('test acc: {}'.format('%.3f' % test_acc))
                return
            cpc.transfer_train(tune_loader, val_loader)
            best_model_dir = os.path.join(cpc.writer.log_dir, 'model_best.pth.tar')
            cpc.test_performance(best_model_dir=best_model_dir, test_loader=test_loader)
        elif args.mol == 'MoCo':
            if args.DAL:
                moco = MoCo(model=model, optimizer=optimizer, scheduler=scheduler, args=args, optimizer_DAL=optimizer_DAL, scheduler_DAL=scheduler_DAL)
            else:
                moco = MoCo(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
            if args.evaluate:
                test_acc, test_f1 = MoCo_evaluate(model=moco.model, criterion=moco.criterion, args=moco.args, data_loader=test_loader)
                print('test f1: {}'.format('%.3f' % test_f1))
                print('test acc: {}'.format('%.3f' % test_acc))
                return
            if args.DAL and args.if_fine_tune:
                moco.transfer_train_DAL(tune_loader, val_loader, train_loader)
            else:
                moco.transfer_train(tune_loader, val_loader)
            best_model_dir = os.path.join(moco.writer.log_dir, 'model_best.pth.tar')
            moco.test_performance(best_model_dir=best_model_dir, test_loader=test_loader)
        else:
            simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
            if args.evaluate:
                test_acc, test_f1 = evaluate(model=simclr.model, criterion=simclr.criterion, args=simclr.args, data_loader=test_loader)
                print('test acc: {}'.format('%.3f' % test_acc))
                return
            simclr.transfer_train(tune_loader, val_loader)
            best_model_dir = os.path.join(simclr.writer.log_dir, 'model_best.pth.tar')
            simclr.test_performance(best_model_dir=best_model_dir, test_loader=test_loader)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
