import logging
import math
import os
import sys
import numpy as np
from os.path import dirname

sys.path.append(dirname(dirname(sys.path[0])))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from judge import AverageMeter
from utils import f1_cal, save_config_file, accuracy, save_checkpoint, evaluate, split_last, merge_last, CPC_evaluate
from sklearn.metrics import f1_score

class ConSSL(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        writer_pos = './runs/' + self.args.store + '/' + self.args.name
        if self.args.transfer is True:
            if self.args.if_fine_tune:
                writer_pos += '_ft'
            else:
                writer_pos += '_le'
            if self.args.shot >= 0:
                writer_pos += f'_shot_{self.args.shot}'
            else:
                writer_pos += f'_percent_{self.args.percent}'
        else:
            writer_pos += '/'
        self.writer = SummaryWriter(writer_pos)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    
    def compute_contrastive_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        logits = logits / self.args.temperature
        return logits, labels


    def train(self, train_loader, val_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start ConSSL training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        acc = 0
        loss = 0
        best_epoch = 0
        best_loss = 1e9

        for epoch_counter in tqdm(range(self.args.epochs)):
            acc_batch = AverageMeter('acc_batch', ':6.2f')
            loss_batch = AverageMeter('loss_batch', ':6.5f')

            self.model.train()
            
            for sensor, _ in train_loader:
                x1 = sensor[:, 0]
                x2 = sensor[:, 1]

                sensor = torch.cat([x1, x2], dim=0).to(self.args.device)
                
                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(sensor)
                    logits, labels = self.compute_contrastive_loss(features)

                    loss = self.criterion(logits, labels)
                
                acc = accuracy(logits, labels, topk=(1,))
                acc_batch.update(acc, sensor.size(0))
                loss_batch.update(loss, sensor.size(0))

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc', acc, global_step=n_iter)
                    self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], global_step=n_iter)
                n_iter += 1
            
            val_acc, val_loss = self.pretrain_evaluate(model=self.model, data_loader=val_loader)
            self.writer.add_scalar('loss_acc', val_acc, global_step=epoch_counter)
            self.writer.add_scalar('loss_eval', val_loss, global_step=epoch_counter)
            
            self.scheduler.step()

            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            if is_best:
                best_epoch = epoch_counter
                # checkpoint_name = 'checkpoint_at_{:04d}.pth.tar'.format(epoch_counter)
                checkpoint_name = 'model_best.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter,
                    'state_dict': self.model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, filename=os.path.join(self.writer.log_dir, checkpoint_name), path_dir=self.writer.log_dir)
            logging.debug(f"Epoch: {epoch_counter} loss: {loss_batch.avg} acc: {acc_batch.avg} eval loss: {val_loss} eval acc :{val_acc}")

        logging.info("Training has finished.")
        logging.info(f"Model of Epoch {best_epoch} checkpoint and metadata has been saved at {self.writer.log_dir}.")

        return
    
    def pretrain_evaluate(self, model, data_loader):
        loss_eval = AverageMeter('Loss', ':.4e')
        acc_eval = AverageMeter('acc_eval', ':6.2f')

        model.eval()

        with torch.no_grad():
            for sensor, _ in data_loader:
                x1 = sensor[:, 0]
                x2 = sensor[:, 1]
                sensor = torch.cat([x1, x2], dim=0).to(self.args.device)
                if sensor.shape == 2:
                    sensor = sensor.unsqueeze(dim=0)

                with autocast(enabled=self.args.fp16_precision):
                    features = model(sensor)
                    logits, labels = self.compute_contrastive_loss(features)

                    loss = self.criterion(logits, labels)

                acc = accuracy(logits, labels, topk=(1,))
                acc_eval.update(acc, sensor.size(0))
                loss_eval.update(loss, sensor.size(0))

        return acc_eval.avg, loss_eval.avg
    

    def transfer_train(self, tune_loader, val_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter_train = 0
        logging.info(f"Start ConSSL fine-tuning head for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        acc = 0
        f1 = 0
        best_epoch = 0
        best_acc = 0
        best_f1 = 0

        for epoch_counter in tqdm(range(self.args.epochs)):
            acc_batch = AverageMeter('acc1', ':6.2f')

            pred_batch = torch.empty(0).to(self.args.device)
            label_batch = torch.empty(0).to(self.args.device)

            if self.args.if_fine_tune:
                self.model.train()
            else: 
                self.model.eval()
                self.model.Classifier.train()

            for sensor, target in tune_loader:

                # sensor = sensor[:, 0].to(self.args.device) ## among the two samples, we only sample the first sample to make the setting aligned with the others.
                sensor = sensor.to(self.args.device)
                target = target[:, 0].to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    logits = self.model(sensor)
                    loss = self.criterion(logits, target)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                acc = accuracy(logits, target, topk=(1,))
                f1 = f1_cal(logits, target, topk=(1,))
                acc_batch.update(acc, sensor.size(0))
                
                label_batch = torch.cat((label_batch, target))
                _, pred = logits.topk(1, 1, True, True)
                pred_batch = torch.cat((pred_batch, pred.reshape(-1)))

                if n_iter_train % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter_train)
                    self.writer.add_scalar('acc', acc, global_step=n_iter_train)
                    self.writer.add_scalar('f1', f1, global_step=n_iter_train)
                    self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], global_step=n_iter_train)

                n_iter_train += 1

            f1_batch = f1_score(label_batch.cpu().numpy(), pred_batch.cpu().numpy(), average='macro') * 100
            val_acc, val_f1 = evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=val_loader)

            is_best = val_f1 > best_f1
            best_f1 = max(val_f1, best_f1)
            best_acc = max(val_acc, best_acc)
            if is_best:
                best_epoch = epoch_counter
                checkpoint_name = 'model_best.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter,
                    'state_dict': self.model.state_dict(),
                    'best_f1': best_f1,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, filename=os.path.join(self.writer.log_dir, checkpoint_name), path_dir=self.writer.log_dir)

            self.scheduler.step()
            self.writer.add_scalar('eval acc', val_acc, global_step=epoch_counter)
            self.writer.add_scalar('eval f1', val_f1, global_step=epoch_counter)
            logging.debug(f"Epoch: {epoch_counter} Loss: {loss} acc: {acc_batch.avg: .3f}/{val_acc: .3f} f1: {f1_batch: .3f}/{val_f1: .3f}")

        logging.info("Fine-tuning has finished.")
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

    

# def cda_evaluate(model, criterion, args, data_loader):
#     losses = AverageMeter('Loss', ':.4e')
#     acc_eval = AverageMeter('acc_eval', ':6.2f')
#     # f1_eval = AverageMeter('f1_eval', ':6.2f')

#     model.eval()

#     label_all = []
#     pred_all = []

#     with torch.no_grad():
#         for sensor, target in data_loader:

#             x1 = sensor[:, 0]
#             x2 = sensor[:, 1]
#             sensor = torch.cat([x1, x2], axis=0).to(args.device)

#             target1 = target[:, 0]
#             target2 = target[:, 0]
#             target = torch.cat([target1, target2], axis=0).to(args.device)

#             with autocast(enabled=args.fp16_precision):
#                 logits = model(sensor)
#                 if type(logits) is tuple:
#                     logits = logits[0]
#                 loss = criterion(logits, target)

#             losses.update(loss.item(), sensor.size(0))
#             _, pred = logits.topk(1, 1, True, True)

#             label_all = np.concatenate((label_all, target.cpu().numpy()))
#             pred_all = np.concatenate((pred_all, pred.cpu().numpy().reshape(-1)))

#             acc = accuracy(logits, target, topk=(1,))
#             acc_eval.update(acc, sensor.size(0))

#     f1_eval = f1_score(label_all, pred_all, average='macro') * 100

#     return acc_eval.avg, f1_eval
