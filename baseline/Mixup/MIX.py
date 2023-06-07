from cProfile import label
from contextlib import AsyncExitStack
import logging
import pdb
from re import T
import sys
from os.path import dirname
from sklearn.metrics import f1_score

sys.path.append(dirname(dirname(sys.path[0])))

import numpy as np
from tqdm import tqdm
from judge import AverageMeter
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from utils import accuracy, evaluate, f1_cal, save_checkpoint, save_config_file

class  MIX(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']

        writer_pos = './runs/' + self.args.store + '/' + self.args.name
        if self.args.shot >=0:
            writer_pos += f'_shot_{self.args.shot}'
        else:
            writer_pos += f'_percent_{self.args.percent}'
        self.writer = SummaryWriter(writer_pos)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def mixup_data(self, x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.args.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)
    
    def train(self, tune_loader, val_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        logging.info(f"Start Mixup training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        best_epoch = 0
        best_f1 = 0

        for epoch_counter in tqdm(range(self.args.epochs)):

            pred_batch = torch.empty(0).to(self.args.device)
            label_a_batch = torch.empty(0).to(self.args.device)
            label_b_batch = torch.empty(0).to(self.args.device)
            loss_batch = 0

            self.model.train()

            for batch_idx, (sensor, target) in enumerate(tune_loader):
                
                sensor, targets_a, targets_b, lam = self.mixup_data(sensor, target[:, 0], alpha=1.0)
                
                sensor = sensor.to(self.args.device)
                targets_a, targets_b = targets_a.to(self.args.device), targets_b.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    logits = self.model(sensor)
                    loss = self.mixup_criterion(logits, targets_a, targets_b, lam)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                loss_batch += loss
                
                label_a_batch, label_b_batch = torch.cat((label_a_batch, targets_a)), torch.cat((label_b_batch, targets_b))
                _, pred = logits.topk(1, 1, True, True)
                pred_batch = torch.cat((pred_batch, pred.reshape(-1)))


            f1_batch = mixup_f1(pred_batch, label_a_batch, label_b_batch, lam)
            acc_batch = mixup_accuracy(pred_batch, label_a_batch, label_b_batch, lam)
            val_acc, val_f1 = evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=val_loader)

            self.writer.add_scalar('loss', loss_batch/(batch_idx+1), global_step=epoch_counter)
            self.writer.add_scalar('acc', acc_batch, global_step=epoch_counter)
            self.writer.add_scalar('f1', f1_batch, global_step=epoch_counter)
            self.writer.add_scalar('eval acc', val_acc, global_step=epoch_counter)
            self.writer.add_scalar('eval f1', val_f1, global_step=epoch_counter)

            is_best = val_f1 > best_f1
            best_f1 = max(val_f1, best_f1)
            if is_best:
                best_epoch = epoch_counter
                checkpoint_name = 'model_best.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter,
                    'state_dict': self.model.state_dict(),
                    'best_f1': best_f1,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, filename=os.path.join(self.writer.log_dir, checkpoint_name), path_dir=self.writer.log_dir)
            
            logging.debug(f"Epoch: {epoch_counter} Loss: {loss_batch/(batch_idx+1)} acc: {acc_batch: .3f}/{val_acc: .3f} f1: {f1_batch: .3f}/{val_f1: .3f}")

        logging.info("Training has finished.")
        logging.info(f"best eval f1 is {best_f1} at {best_epoch}.")

        print('best eval f1 is {} for {}'.format(best_f1, self.args.name))

    def test_performance(self, best_model_dir, test_loader):
        checkpoint = torch.load(best_model_dir, map_location="cpu")
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        test_acc, test_f1 = evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=test_loader)
        logging.info(f"test f1 is {test_f1}.")
        logging.info(f"test acc is {test_acc}.")
        
        print('test f1 is {} for {}'.format(test_f1, self.args.name))
        print('test acc is {} for {}'.format(test_acc, self.args.name))
    
def mixup_accuracy(predicted, y_a, y_b, lam):
    with torch.no_grad():
        correct = (lam * predicted.eq(y_a).cpu().sum().float()
                        + (1 - lam) * predicted.eq(y_b).cpu().sum().float())
    return 100 * correct / y_a.shape[0]

def mixup_f1(pred, y_a, y_b, lam):
    with torch.no_grad():
        f1_a = f1_score(y_a.cpu().numpy(), pred.cpu().numpy(), average='macro') * 100
        f1_b = f1_score(y_b.cpu().numpy(), pred.cpu().numpy(), average='macro') * 100
    return lam * f1_a + (1 - lam) * f1_b
