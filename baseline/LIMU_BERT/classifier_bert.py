#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : classifier_bert.py
# @Description :
import argparse
import pdb
import sys
from os.path import dirname
sys.path.append(dirname(dirname(sys.path[0])))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from data_aug.contrastive_learning_dataset import GenerativeDataset


import train
from config import load_dataset_label_names
from models import BERTClassifier, fetch_classifier

from statistic import stat_acc_f1
from utils import Preprocess4Tensor, get_device,  handle_argv \
    , load_bert_classifier_data_config, Preprocess4Normalization, seed_torch
from torchvision.transforms import transforms
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

## frozen_bert
def bert_classify(args, frozen_bert=False, balance=True):
    train_cfg, model_bert_cfg, model_classifier_cfg = load_bert_classifier_data_config(args)
    seed_torch(args.seed)

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_index}')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    pipeline = transforms.Compose([Preprocess4Normalization(model_bert_cfg.feature_num, dataset_name=args.name),
                                  Preprocess4Tensor()])  # the second is to transform data to tensor(dtype=float).
    dataset = GenerativeDataset(pipeline=pipeline, version=args.dataset_version, datasets_name=args.name)

    tune_dataset = dataset.get_dataset(split='tune', percent=args.percent, shot=args.shot)
    val_dataset = dataset.get_dataset(split='val')
    test_dataset = dataset.get_dataset(split='test')
    
    tune_loader = DataLoader(tune_dataset, batch_size=train_cfg.batch_size, shuffle=True, pin_memory=False, drop_last=False, num_workers=5)  # make it consistant to Ours
    if len(val_dataset) < train_cfg.batch_size:
        val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=True, pin_memory=False, drop_last=False, num_workers=5)  # make it consistant to Ours
    else:
        val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=True, pin_memory=False, drop_last=True, num_workers=5)  # make it consistant to Ours
    test_loader = DataLoader(test_dataset, batch_size=train_cfg.batch_size, shuffle=True, pin_memory=False, drop_last=False, num_workers=5)
    
    criterion = nn.CrossEntropyLoss()
    classifier = fetch_classifier(method, model_classifier_cfg, input=model_bert_cfg.hidden, dataset_name=args.name)
    model = BERTClassifier(model_bert_cfg, classifier=classifier, frozen_bert=frozen_bert)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    trainer = train.Trainer(args, train_cfg, model, optimizer, args.save_path, args.device)

    def func_loss(model, batch): 
        inputs, label = batch
        logits = model(inputs, True)
        loss = criterion(logits, label)
        return loss

    def func_forward(model, batch):
        inputs, label = batch
        logits = model(inputs, False)
        return logits, label

    def func_evaluate(label, predicts):
        stat = stat_acc_f1(label.cpu().numpy(), predicts.cpu().numpy())
        return stat
    with torch.cuda.device(args.gpu_index):
        trainer.train(func_loss, func_forward, func_evaluate, tune_loader, test_loader, val_loader
                            , model_file=args.pretrain_model, load_self=True)
        trainer.run(func_forward, None, test_loader)
    return


if __name__ == "__main__":
    balance = True
    method = "base_gru"
    args = handle_argv('bert_classifier_' + method, 'bert_classifier_train.json', method)
    with torch.cuda.device(args.gpu_index):
        bert_classify(args, frozen_bert=args.frozen_bert, balance=balance)


