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

SMALL_NUM = np.log(1e-45)


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.encoder = Convolutional1DEncoder(args)

    def forward(self, inputs):
        return self.encoder(inputs)


class Convolutional1DEncoder(nn.Module):
    def __init__(self, args):
        super(Convolutional1DEncoder, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(args.input_size, 32, kernel_size=args.kernel_size,
                      stride=1, padding=args.padding,
                      padding_mode='reflect'),
            ConvBlock(32, 64, kernel_size=args.kernel_size,
                      stride=1, padding=args.padding,
                      padding_mode='reflect'),
            ConvBlock(64, 128, kernel_size=args.kernel_size,
                      stride=1, padding=args.padding,
                      padding_mode='reflect')
        )

    def forward(self, inputs):
        # Tranposing since the Conv1D requires
        inputs = inputs.permute(0, 2, 1)
        encoder = self.encoder(inputs)
        encoder = encoder.permute(0, 2, 1)

        return encoder


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=1, padding_mode='reflect', dropout_prob=0.2):
        super(ConvBlock, self).__init__()

        # 1D convolutional layer
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              padding_mode=padding_mode,
                              bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, inputs):
        conv = self.conv(inputs)
        relu = self.relu(conv)
        dropout = self.dropout(relu)

        return dropout

class CPC_autoregressor(nn.Module):
    def __init__(self):
         super().__init__()
         self.rnn = nn.GRU(input_size=128,
                           hidden_size=256,
                           num_layers=2,
                           bidirectional=False,
                           batch_first=True,
                           dropout=0.2)

    def forward(self, x):
        x, _ = self.rnn(x)
        return x


class Classifier(nn.Module):
    def __init__(self, classes):
        super(Classifier, self).__init__()
        # Softmax
        self.softmax = nn.Sequential(nn.Linear(256, 256),
                                     nn.BatchNorm1d(256),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.2),
                                     nn.Linear(256, 128),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.2),
                                     nn.Linear(128, classes))

    def forward(self, inputs):
        softmax = self.softmax(inputs[:, -1, :])
        return softmax

class Transfer_Coder(nn.Module):
    def __init__(self, classes, args):
        super(Transfer_Coder, self).__init__()
        self.encoder = Encoder(args)
        self.ar = CPC_autoregressor()
        self.Classifier = Classifier(classes)

    
    def forward(self, x):
        h = self.encoder(x)
        h = self.ar(h)
        z = self.Classifier(h)
        return z


class CPC_model(nn.Module):
    def __init__(self, args):

        super(CPC_model, self).__init__()
        self.transfer = args.transfer
        self.batch_size = args.batch_size
        self.num_steps_prediction = args.num_steps_prediction
        self.device = args.device

        self.encoder = Encoder(args)
        self.ar = CPC_autoregressor()
        
        if self.transfer:
            self.classifier = Classifier(classes=args.classes)

        self.Wk = nn.ModuleList([PredictionNetwork()
                                 for i in range(self.num_steps_prediction)])
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        # Passing through the encoder. Input: BxCxT and output is: Bx128XT
        z = self.encoder(x)

        # Random timestep to start the future prediction from.
        # If the window is 50 timesteps and k=12, we pick a number from 0-37
        start = torch.randint(int(x.shape[1] - self.num_steps_prediction),
                              size=(1,)).long()

        # Need to pick the encoded data only until the starting timestep
        rnn_input = z[:, :start + 1, :]

        # Passing through the RNN
        r_out = self.ar(rnn_input)

        accuracy, nce, correct_steps = self.compute_cpc_loss(z, r_out, start)

        return accuracy, nce, correct_steps

    def compute_cpc_loss(self, z, c, t):
        batch_size = z.shape[0]

        # The context vector is the last timestep from the RNN
        c_t = c[:, t, :].squeeze(1)

        # infer z_{t+k} for each step in the future: c_t*Wk, where 1 <= k <=
        # timestep
        pred = torch.stack([self.Wk[k](c_t) for k in
                            range(self.num_steps_prediction)])

        # pick the target z values k timestep number of samples after t
        z_samples = z[:, t + 1: t + 1 + self.num_steps_prediction, :] \
            .permute(1, 0, 2)

        nce = 0
        correct = 0
        correct_steps = []

        # Looping over the number of timesteps chosen
        for k in range(self.num_steps_prediction):
            # calculate the log density ratio: log(f_k) = z_{t+k}^T * W_k * c_t
            log_density_ratio = torch.mm(z_samples[k], pred[k].transpose(0, 1))

            # correct if highest probability is in the diagonal
            positive_batch_pred = torch.argmax(self.softmax(log_density_ratio),
                                               dim=0)
            positive_batch_actual = torch.arange(0, batch_size).to(self.device)
            correct = (correct +
                       torch.sum(torch.eq(positive_batch_pred,
                                          positive_batch_actual)).item())
            correct_steps.append(torch.sum(torch.eq(positive_batch_pred,
                                                    positive_batch_actual)).item())

            # calculate NCE loss
            nce = nce + torch.sum(torch.diag(self.lsoftmax(log_density_ratio)))

        # average over timestep and batch
        nce = nce / (-1.0 * batch_size * self.num_steps_prediction)
        accuracy = correct / (1.0 * batch_size * self.num_steps_prediction) * 100
        correct_steps = np.array(correct_steps)
        return accuracy, nce, correct_steps

    def predict_features(self, inputs):
        z = self.encoder(inputs)

        # Passing through the RNN
        r_out = self.ar(z)

        return r_out
    

class PredictionNetwork(nn.Module):
    def __init__(self):
        super(PredictionNetwork, self).__init__()
        self.Wk = nn.Linear(256, 128)

    def forward(self, inputs):
        prediction = self.Wk(inputs)

        return prediction


class CPC(object):

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
            if self.args.shot:
                writer_pos += f'_shot_{self.args.shot}'
            else:
                writer_pos += f'_percent_{self.args.percent}'
        else:
            writer_pos += '/'
        self.writer = SummaryWriter(writer_pos)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def train(self, train_loader, val_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start CPCHAR training for {self.args.epochs} epochs.")
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
                sensor = sensor.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    acc, loss, _ = self.model(sensor)

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

    def pretrain_evaluate(self, model, data_loader):
        loss_eval = AverageMeter('Loss', ':.4e')
        acc_eval = AverageMeter('acc_eval', ':6.2f')

        model.eval()

        with torch.no_grad():
            for sensor, target in data_loader:
                sensor = sensor.to(self.args.device)
                if sensor.shape == 2:
                    sensor = sensor.unsqueeze(dim=0)
                target = target[:, 0].to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    acc, loss, _ = model(sensor)
                
                acc_eval.update(acc, sensor.size(0))
                loss_eval.update(loss, sensor.size(0))

        return acc_eval.avg, loss_eval.avg
    
    def transfer_train(self, tune_loader, val_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter_train = 0
        logging.info(f"Start CPC fine-tuning head for {self.args.epochs} epochs.")
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
