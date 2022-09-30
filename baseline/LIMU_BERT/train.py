#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:22
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : train.py
# @Description :
from cProfile import label
import copy
import logging
import os
import sys
from os.path import dirname
from tqdm import tqdm
import pdb
sys.path.append(dirname(dirname(sys.path[0])))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import count_model_parameters


class Trainer(object):
    def __init__(self, args, cfg, model, optimizer, save_path, device):
        self.args = args
        self.cfg = cfg # config for training : see class Config
        self.model = model
        self.optimizer = optimizer
        self.save_path = save_path
        self.device = device # device name

        writer_pos = self.save_path
        self.writer = SummaryWriter(writer_pos)
        # to record data
        
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        

    def pretrain(self, func_loss, func_forward, func_evaluate
              , data_loader_train, data_loader_test, model_file=None, data_parallel=False):
        """ Train Loop """
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)
        
        best_epoch = 0
        global_step = 0 # global iteration steps regardless of epochs
        best_loss = 1e6
        loss_eva = 0
        model_best = model.state_dict()

        logging.info(f"Start LIMU training for {self.cfg.n_epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")

        for e in tqdm(range(self.cfg.n_epochs)):
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            self.model.train()
            for batch in data_loader_train:
                batch = [t.to(self.device) for t in batch]
                self.optimizer.zero_grad()
                loss = func_loss(model, batch)

                loss = loss.mean()# mean() for Data Parallelism
                loss.backward()
                self.optimizer.step()

                if global_step % self.args.log_every_n_steps == 0 and global_step != 0:
                    self.writer.add_scalar('loss', loss, global_step=global_step)
                    self.writer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], global_step=global_step)

                global_step += 1
                loss_sum += loss.item()

                # if global_step % self.cfg.save_steps == 0: # save
                #     self.save(global_step)

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('The Total Steps have been reached.')
                    return

            loss_eva = self.run(func_forward, func_evaluate, data_loader_test)
            self.writer.add_scalar('eval loss', loss_eva, global_step=global_step)
            logging.debug(f"Epoch: {e} \t Loss: {loss} \t loss eva: {loss_eva} \t")

            if loss_eva < best_loss:
                best_epoch = e
                best_loss = loss_eva
                model_best = copy.deepcopy(model.state_dict())
                self.save(0)
        
        print('best eval loss is {}'.format(best_loss))
        logging.info("Fine-tuning has finished.")
        logging.info(f"best eval loss is {loss_eva} at {best_epoch}.")

        model.load_state_dict(model_best)

    def run(self, func_forward, func_evaluate, data_loader, model_file=None, data_parallel=False, load_self=False):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        self.load(model_file, load_self=load_self)
        # print(count_model_parameters(self.model))
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        results = [] # prediction results
        labels = []
        for batch in data_loader:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                result, label = func_forward(model, batch)
                results.append(result)
                labels.append(label)
        if func_evaluate:
            return func_evaluate(torch.cat(labels, 0), torch.cat(results, 0))
        else:
            return torch.cat(results, 0).cpu().numpy()

    def train(self, func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, data_loader_vali
              , model_file=None, data_parallel=False, load_self=False):
        """ Train Loop """
        self.load(model_file, load_self)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0 # global iteration steps regardless of epochs
        eval_acc_best = 0.0
        best_epoch = 0
        best_stat = None
        model_best = model.state_dict()

        logging.info(f"Start LIMU fine-tuning head for {self.cfg.n_epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")


        for e in tqdm(range(self.cfg.n_epochs)):
            loss_sum = 0.0 # the sum of iteration losses to get average loss in every epoch
            self.model.train()
            for batch in data_loader_train:
                batch = [t.to(self.device) for t in batch]

                self.optimizer.zero_grad()
                loss = func_loss(model, batch)

                loss = loss.mean()# mean() for Data Parallelism
                loss.backward()
                self.optimizer.step()

                if global_step % self.args.log_every_n_steps == 0 and global_step != 0:
                        self.writer.add_scalar('loss', loss, global_step=global_step)
                        self.writer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], global_step=global_step)

                global_step += 1
                loss_sum += loss.item()
                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('The Total Steps have been reached.')
                    return
                
            train_acc, train_f1 = self.run(func_forward, func_evaluate, data_loader_train)
            test_acc, test_f1 = self.run(func_forward, func_evaluate, data_loader_test)
            eval_acc, eval_f1 = self.run(func_forward, func_evaluate, data_loader_vali)
            
            if eval_acc > eval_acc_best:
                best_epoch = e
                eval_acc_best = eval_acc
                best_stat = (train_acc, eval_acc, test_acc, train_f1, eval_f1, test_f1)
                model_best = copy.deepcopy(model.state_dict())
                self.save(0)
            
            self.writer.add_scalar('train acc', train_acc, global_step=global_step)
            self.writer.add_scalar('eval acc', eval_acc, global_step=global_step)
            self.writer.add_scalar('test acc', test_acc, global_step=global_step)
            self.writer.add_scalar('train f1', train_f1, global_step=global_step)
            self.writer.add_scalar('eval f1', eval_f1, global_step=global_step)
            self.writer.add_scalar('test f1', test_f1, global_step=global_step)
            
            logging.debug(f"Epoch: {e} \t Loss: {loss} \t train acc: {train_acc} \t eval acc: {eval_acc}"+
                          f"test acc: {test_acc} \t train f1: {train_f1} \t eval f1: {eval_f1} \t test f1: {test_f1}")
        
        self.model.load_state_dict(model_best)

        print(f'Best Accuracy: {best_stat[0]:0.4f} / {best_stat[1]:.4f} / {best_stat[2]:.4f}, F1: {best_stat[3]:.4f} / {best_stat[4]:.4f} / {best_stat[5]:.4f}')
        logging.info("Fine-tuning has finished.")
        logging.info(f"best eval acc is {eval_acc_best} at {best_epoch}.")


    def load(self, model_file, load_self=False):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            if load_self:
                self.model.load_self(model_file + '.pt', map_location=self.device)
            else:
                self.model.load_state_dict(torch.load(model_file + '.pt', map_location=self.device))

    def save(self, i=0):
        """ save current model """
        if i != 0:
            torch.save(self.model.state_dict(), self.save_path + "_" + str(i) + '.pt')
        else:
            torch.save(self.model.state_dict(), self.save_path + '.pt')

