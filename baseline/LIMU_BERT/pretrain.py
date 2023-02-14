#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : pretrain.py
# @Description :
import argparse
from cProfile import label
from cgi import test
from random import seed
import sys
from os.path import dirname
sys.path.append(dirname(dirname(sys.path[0])))


import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import models, train
from config import MaskConfig, TrainConfig, PretrainModelConfig
from data_aug.contrastive_learning_dataset import GenerativeDataset, LIMUDataset4Training
from models import LIMUBertModel4Pretrain
from utils import seed_torch, get_device, handle_argv, load_pretrain_data_config, Preprocess4Normalization,  Preprocess4Mask
from torchvision.transforms import transforms
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

torch.set_default_tensor_type(torch.DoubleTensor)

def main(args):
    train_cfg, model_cfg, mask_cfg = load_pretrain_data_config(args)
    seed_torch(args.seed)

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_index}')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    pipeline = transforms.Compose([Preprocess4Normalization(model_cfg.feature_num, dataset_name=args.name), Preprocess4Mask(mask_cfg)])

    dataset = GenerativeDataset(pipeline=pipeline, version=args.dataset_version, datasets_name=args.name)
    train_dataset = dataset.get_dataset(split='train')
    val_dataset = dataset.get_dataset(split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, pin_memory=False, drop_last=True, num_workers=5)  # make it consistant to Ours
    if len(val_dataset) < train_cfg.batch_size:
        val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=True, pin_memory=False, drop_last=False, num_workers=5)
    else:
        val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=True, pin_memory=False, drop_last=True, num_workers=5)  # make it consistant to Ours

    model = LIMUBertModel4Pretrain(model_cfg)
    model = model.float()

    criterion = nn.MSELoss(reduction='none')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    trainer = train.Trainer(args, train_cfg, model, optimizer, args.save_path, args.device)

    def func_loss(model, batch):
        mask_seqs, masked_pos, seqs, _ = batch #

        seq_recon = model(mask_seqs, masked_pos) #
        loss_lm = criterion(seq_recon, seqs) # for masked LM
        return loss_lm

    def func_forward(model, batch):
        mask_seqs, masked_pos, seqs, _ = batch
        seq_recon = model(mask_seqs, masked_pos)
        return seq_recon, seqs

    def func_evaluate(seqs, predict_seqs):
        loss_lm = criterion(predict_seqs, seqs)
        return loss_lm.mean().cpu().numpy()

    if hasattr(args, 'pretrain_model'):
        trainer.pretrain(func_loss, func_forward, func_evaluate, train_loader, val_loader, model_file=args.pretrain_model)
    else:
        trainer.pretrain(func_loss, func_forward, func_evaluate, train_loader, val_loader, model_file=None)


if __name__ == "__main__":
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    main(args)
