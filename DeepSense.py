import pdb
from traceback import print_tb
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np
from tqdm import tqdm
from judge import AverageMeter
import os
from utils import accuracy, evaluate, f1_cal, save_checkpoint, save_config_file

CONV_LEN = 3
CONV_LEN_INTE = 3#4
CONV_LEN_LAST = 3#5
CONV_NUM = 64
CONV_MERGE_LEN = 8
CONV_MERGE_LEN2 = 6
CONV_MERGE_LEN3 = 4
CONV_NUM2 = 64
INTER_DIM = 120

class DeepSense_model(nn.Module):
    def __init__(self, classes=6, dims=64):
        super(DeepSense_model, self).__init__()

        self.encoder = DeepSense_encoder(dims=dims)
        self.classifier = nn.Linear(in_features=120, out_features=classes)

    def forward(self, x):
        h = self.encoder(x)
        h = torch.mean(h, dim=1)
        h = self.classifier(h)
        return h


class DeepSense_encoder(nn.Module):

    def __init__(self, dims=64):
        super(DeepSense_encoder, self).__init__()

        self.dropout = torch.nn.Dropout(p=0.2)
        self.relu = torch.nn.ReLU()

        # (N, Channel=1, H, W)
        self.conv1_acc = torch.nn.Conv2d(1, dims, kernel_size=(2*3*CONV_LEN, 1), stride=[1, 2*3])
        self.conv1_acc_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)
        
        self.conv2_acc = torch.nn.Conv2d(dims, dims, kernel_size=(CONV_LEN_INTE, 1), stride=[1, 1])
        self.conv2_acc_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)

        self.conv3_acc = torch.nn.Conv2d(dims, dims, kernel_size=(CONV_LEN_LAST, 1), stride=[1, 1])
        self.conv3_acc_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)

        self.conv1_gyro = torch.nn.Conv2d(1, dims, kernel_size=(2*3*CONV_LEN, 1), stride=[1, 2*3])
        self.conv1_gyro_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)
        
        self.conv2_gyro = torch.nn.Conv2d(dims, dims, kernel_size=(CONV_LEN_INTE, 1), stride=[1, 1])
        self.conv2_gyro_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)

        self.conv3_gyro = torch.nn.Conv2d(dims, dims, kernel_size=(CONV_LEN_LAST, 1), stride=[1, 1])
        self.conv3_gyro_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)

        self.conv1_sensor = torch.nn.Conv2d(dims, dims, kernel_size=(CONV_MERGE_LEN, 1), stride=[1, 1])
        self.conv1_sensor_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)
        
        self.conv2_sensor = torch.nn.Conv2d(dims, dims, kernel_size=(CONV_MERGE_LEN2, 1), stride=[1, 1])
        self.conv2_sensor_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)

        self.conv3_sensor = torch.nn.Conv2d(dims, dims, kernel_size=(CONV_MERGE_LEN3, 1), stride=[1, 1])
        self.conv3_sensor_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)
        
        self.gru1 = torch.nn.GRU(dims, INTER_DIM, num_layers=1, batch_first=True, bidirectional=False)
        self.gru2 = torch.nn.GRU(INTER_DIM, INTER_DIM, num_layers=1, batch_first=True, bidirectional=False)
        
    
    def forward(self, x):
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()

        # data shape: (BATCH_SIZE, CHANNEL=1, Timestamps, FEATURE_DIM)
        h_acc = self.conv1_acc_BN(self.conv1_acc(x[:, :, :, 0:3]))
        h_acc = self.dropout(self.relu(h_acc))

        h_acc = self.conv2_acc_BN(self.conv2_acc(h_acc))
        h_acc = self.dropout(self.relu(h_acc))

        h_acc = self.conv3_acc_BN(self.conv3_acc(h_acc))
        h_acc = self.relu(h_acc)
        h_acc_shape = h_acc.shape
        h_acc_out = h_acc.reshape(h_acc_shape[0], h_acc_shape[1], -1, 1)
        
        h_gyro = self.conv1_gyro_BN(self.conv1_gyro(x[:, :, :, 3:]))
        h_gyro = self.dropout(self.relu(h_gyro))

        h_gyro = self.conv2_gyro_BN(self.conv2_gyro(h_gyro))
        h_gyro = self.dropout(self.relu(h_gyro))

        h_gyro = self.conv3_gyro_BN(self.conv3_gyro(h_gyro))
        h_gyro = self.relu(h_gyro)
        h_gyro_shape = h_gyro.shape
        h_gyro_out = h_gyro.reshape(h_gyro_shape[0], h_gyro_shape[1], -1, 1)
        
        
        # reshape to (BATCH_SIZE, Channels, FEATURE_DIM, 2)
        h = torch.cat((h_acc_out, h_gyro_out), dim=3)
        h = self.dropout(h)

        h = self.conv1_sensor_BN(self.conv1_sensor(h))
        h = self.dropout(self.relu(h))

        h = self.conv2_sensor_BN(self.conv2_sensor(h))
        h = self.dropout(self.relu(h))

        h = self.conv3_sensor_BN(self.conv3_sensor(h))
        h = self.relu(h)

        # reshape to (BATCH_SIZE, FEATURE_DIM, 2)
        h = h.flatten(start_dim=2)
        h_shape = h.shape
        h = h.reshape(h.shape[0], h_shape[2], -1)
        h, _ = self.gru1(h)
        h = self.dropout(h)
        h, _ = self.gru2(h)
        h = self.dropout(h)
        return h


class DeepSense(object):

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

    def train(self, tune_loader, val_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        logging.info(f"Start DeepSense training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        best_epoch = 0
        best_f1 = 0

        for epoch_counter in tqdm(range(self.args.epochs)):

            logits_batch = torch.empty(0).to(self.args.device)
            label_batch = torch.empty(0).to(self.args.device)
            loss_batch = 0

            self.model.train()

            for batch_idx, (sensor, target) in enumerate(tune_loader):

                sensor = sensor.to(self.args.device)
                targets = target[:, 0].to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    logits = self.model(sensor)
                    loss = self.criterion(logits, targets)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                loss_batch += loss
                
                label_batch = torch.cat((label_batch, targets))
                logits_batch = torch.cat((logits_batch, logits))


            f1_batch = f1_cal(logits_batch, label_batch, topk=(1,))
            acc_batch = accuracy(logits_batch, label_batch, topk=(1,))
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


