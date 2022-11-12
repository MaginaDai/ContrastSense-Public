import code
import logging
import pdb
import sys
from os.path import dirname
from numpy import pad
sys.path.append(dirname(dirname(sys.path[0])))

from tqdm import tqdm
from judge import AverageMeter
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from utils import accuracy, evaluate, f1_cal, save_checkpoint, save_config_file



WINDOW_LENGTH = 200

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        # we could not keep the window size all the same but try to avoid a large difference with the original implementation.
        self.convA = nn.Conv1d(3, 24, kernel_size=10, padding=4)
        self.Prelu_A = nn.PReLU()
        self.LN_A = nn.LayerNorm([24, 199], eps=1e-3)

        self.convB = nn.Conv1d(24, 48, kernel_size=8, padding=3)
        self.Prelu_B = nn.PReLU()
        self.LN_B = nn.LayerNorm([48, 198], eps=1e-3)

        self.convC = nn.Conv1d(48, 20, kernel_size=4, padding=2)
        self.Prelu_C = nn.PReLU()
        self.BN_C = nn.BatchNorm1d(20)

        self.linear = nn.Linear(199*20, 20)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h = self.Prelu_A(self.LN_A(self.convA(x)))
        h = self.Prelu_B(self.LN_B(self.convB(h)))
        h = self.Prelu_C(self.convC(h))

        h = self.BN_C(h)
        h = torch.flatten(h, start_dim=1)
        h = self.linear(h)
        return h


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

    def forward(self, h):
        
        return h

class COCOA_model(nn.Module):
    def __init__(self):
        super(COCOA_model, self).__init__()
        self.acc_encoder = Encoder()
        self.gyro_encoder = Encoder()

    def forward(self, x):
        acc, gyro = x[:, :, 0:3], x[:, :, 3:]
        h_acc = self.acc_encoder(acc)
        h_gyro = self.gyro_encoder(gyro)
        return h_acc, h_gyro


class COCOA(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        writer_pos = './runs/' + self.args.store + '/'
        if self.args.transfer is True:
            if self.args.if_fine_tune:
                writer_pos += self.args.name + '_ft'
            else:
                writer_pos += self.args.name + '_le'
            if self.args.shot:
                writer_pos += f'_shot_{self.args.shot}'
            else:
                writer_pos += f'_percent_{self.args.percent}'
        self.writer = SummaryWriter(writer_pos)

        self.softmax = nn.Softmax(dim=0)

        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
    
    def COCOA_loss(self, h_acc, h_gyro):

        h_acc = F.normalize(h_acc, dim=1)
        h_gyro = F.normalize(h_gyro, dim=1)
        
        h = torch.stack([h_acc, h_gyro], dim=2)
        
        batch_size = h.shape[0]
        dim_size = h.shape[1]

        pos_error = torch.Tensor([0]).to(self.args.device)
        neg_error = torch.Tensor([0]).to(self.args.device)
        for i in range(batch_size):
            sim = torch.matmul(h[i], h[i].T)
            sim = torch.ones([dim_size, dim_size]).to(self.args.device) - sim
            sim = torch.exp(sim/self.args.temperature)
            pos_error += torch.mean(sim)
        
        mask = torch.eye(batch_size, dtype=torch.bool).to(self.args.device)

        for i in range(dim_size):
            sim = torch.matmul(h[:, i, :], h[:, i, :].T)
            sim = torch.exp(sim/self.args.temperature)
            neg_error += torch.mean(sim[~mask.bool()])
        
        loss = pos_error / batch_size + self.args.weight * neg_error/dim_size
        
        with torch.no_grad():
            similarity_matrix = torch.matmul(h_acc, h_gyro.T)
            positive_predict = torch.argmax(self.softmax(similarity_matrix), dim=0)
            positive_actual = torch.arange(0, h_acc.shape[0]).to(self.args.device)
            accuracy = torch.sum(torch.eq(positive_predict, positive_actual)).item() / h_acc.shape[0] * 100
        return loss, accuracy


    def COCOA_loss_mine(self, h_acc, h_gyro):
        h_acc = F.normalize(h_acc, dim=1)
        h_gyro = F.normalize(h_gyro, dim=1)

        similarity_matrix = torch.matmul(h_acc, h_gyro.T)
        
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(h_acc.shape[0], dtype=torch.bool).to(self.args.device)

        # select and combine multiple positives
        positives = similarity_matrix[mask.bool()].view(h_acc.shape[0], -1)
        logits_c = torch.ones([h_acc.shape[0], 1]).to(self.args.device) - positives
        loss_c = torch.mean(torch.exp(logits_c / self.args.temperature))

        # select only the negatives the negatives
        negatives = similarity_matrix[~mask.bool()].view(similarity_matrix.shape[0], -1)
        loss_d = torch.mean(torch.exp(negatives / self.args.temperature))
        loss = loss_c + self.args.weight * loss_d

        positive_predict = torch.argmax(self.softmax(similarity_matrix), dim=0)
        positive_actual = torch.arange(0, h_acc.shape[0]).to(self.args.device)

        accuracy = torch.sum(torch.eq(positive_predict, positive_actual)).item() / h_acc.shape[0] * 100

        return loss, accuracy

    
    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start COCOA training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        acc = 0
        best_epoch = 0
        best_acc = 0
        not_best_counter = 0
        
        for epoch_counter in tqdm(range(self.args.epochs)):
            acc_batch = AverageMeter('acc_batch', ':6.2f')
            loss_batch = AverageMeter('loss_batch', ':6.5f')
            for sensor, _ in train_loader:
                sensor = sensor.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    feature_acc, feature_gyro = self.model(sensor)
                    loss, acc = self.COCOA_loss(feature_acc, feature_gyro)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                
                acc_batch.update(acc, sensor[0].size(0))
                loss_batch.update(loss, sensor[0].size(0))

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc', acc, global_step=n_iter)

                n_iter += 1
            
            is_best = True # Save loss term. 
            if is_best:
                best_epoch = epoch_counter
                checkpoint_name = 'model_best.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter,
                    'state_dict': self.model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, filename=os.path.join(self.writer.log_dir, checkpoint_name), path_dir=self.writer.log_dir)

            log_str = f"Epoch: {epoch_counter} Loss: {loss_batch.avg} accuracy: {acc_batch.avg}"
            logging.debug(log_str)
            
        logging.info("Training has finished.")
        logging.info(f"Model of Epoch {best_epoch} checkpoint and metadata has been saved at {self.writer.log_dir}.")


    def transfer_train(self, tune_loader, val_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter_train = 0
        logging.info(f"Start COCOA fine-tuning head for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        acc = 0
        f1 = 0
        best_epoch = 0
        best_acc = 0
        best_f1 = 0

        for epoch_counter in tqdm(range(self.args.epochs)):
            acc_batch = AverageMeter('acc_batch', ':6.2f')
            f1_batch = AverageMeter('f1_batch', ':6.2f')
            if self.args.if_fine_tune:
                self.model.train()
            else: 
                self.model.eval()
                self.model.Classifier.train()

            for sensor, target in tune_loader:

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
                f1_batch.update(f1, sensor.size(0))
                if n_iter_train % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter_train)
                    self.writer.add_scalar('acc', acc, global_step=n_iter_train)
                    self.writer.add_scalar('f1', f1, global_step=n_iter_train)
                    self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], global_step=n_iter_train)

                n_iter_train += 1

            val_acc, val_f1 = evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=val_loader)

            is_best = val_f1 > best_f1
            best_f1 = max(val_f1, best_acc)
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

            self.writer.add_scalar('eval acc', val_acc, global_step=epoch_counter)
            self.writer.add_scalar('eval f1', val_f1, global_step=epoch_counter)
            self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter} Loss: {loss} acc: {acc_batch.avg: .3f}/{val_acc: .3f} f1: {f1_batch.avg: .3f}/{val_f1: .3f}")
        
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