import sys
import os
import logging
from os.path import dirname
sys.path.append(dirname(dirname(sys.path[0])))

import numpy as np
import torch
import torch.nn as nn

from scipy import signal
from MoCo import GradientReversal
from tqdm import tqdm
from judge import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils import accuracy, evaluate, f1_cal, save_checkpoint, save_config_file

class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError

class Preprocess4STFT(Pipeline):

    def __init__(self, window=50, cut_off_frequency=17, fs=20):
        super().__init__()
        self.window = window
        self.cut_off_frequency = cut_off_frequency
        self.fs = fs

    def __call__(self, instance):
        instance_new = []
        for i in range(instance.shape[1]):
            f, t, Zxx = signal.stft(instance[:, i], self.fs, nperseg=self.window)
            instance_new.append(Zxx[:self.cut_off_frequency, :])
        instance_new = np.abs(np.vstack(instance_new).transpose([1, 0]))
        return instance_new


alpha=1.0
beta=1.0



class BenchmarkXHAR(nn.Module):

    def __init__(self, training=False, lam=1.0, classes=6):
        super().__init__()

        self.training = training
        self.lam = lam

        self.feature = nn.Sequential(nn.Conv1d(input, 64, kernel_size=2),
                                     nn.BatchNorm1d(64),
                                     nn.MaxPool2d(2),
                                     nn.ReLU(True))

        self.lstm = nn.GRU(input - 1, 100, num_layers=3, batch_first=True, bidirectional=True)

        self.class_classifier = nn.Sequential(nn.Linear(464, 200),
                                              nn.BatchNorm1d(200),
                                              nn.ReLU(True),
                                              nn.Dropout2d(),
                                              nn.Linear(200, 100),
                                              nn.BatchNorm1d(100),
                                              nn.ReLU(True),
                                              nn.Linear(100, classes),
                                            )
        

        self.domain_classifier = nn.Sequential(GradientReversal(),
                                               nn.Linear(464, 100),
                                               nn.BatchNorm1d(100),
                                               nn.ReLU(True),
                                               nn.Linear(100, 2))

        self.cnn_domain_classifier = nn.Sequential(GradientReversal(),
                                                   nn.Linear(64, 10),
                                                   nn.BatchNorm1d(10),
                                                   nn.ReLU(True),
                                                   nn.Linear(10, 2))

        self.gru_domain_classifier = nn.Sequential(GradientReversal(),
                                                   nn.Linear(400, 100),
                                                   nn.BatchNorm1d(100),
                                                   nn.ReLU(True),
                                                   nn.Linear(100, 2))

    '''
     input_data: [batch_size, time, dim]  [32, 4, 51]
    '''

    def forward(self, input_data, training=False, lam=1.0):
        feature_gru, _ = self.lstm(input_data[:, :, :-1])  # 32,4,100

        f_head = feature_gru[:, 0, :]  # 32,1,100
        f_tail = feature_gru[:, feature_gru.shape[1] - 1, :]  # 32,1,100
        feature_gru_cat = torch.cat((f_head, f_tail), dim=-1)  # 32,200

        feature_cnn = self.feature(torch.transpose(input_data, 1, 2))  # batch_size, 32, 1
        feature_cnn = feature_cnn.reshape((feature_cnn.shape[0], -1))  # batch_size, 32
        feature_cat = torch.cat((feature_gru_cat, feature_cnn), dim=-1)  # # batch_size, 32+2*100

        class_output = self.class_classifier(feature_cat)

        if training is True:
            gru_domain_classifier = self.gru_domain_classifier(feature_gru_cat)
            cnn_domain_classifier = self.cnn_domain_classifier(feature_cnn)
            domain_output = self.domain_classifier(feature_cat)
            return class_output, domain_output, gru_domain_classifier, cnn_domain_classifier
        else:
            return class_output


class XHAR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
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
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    
    def loss_xhar(self, inputs_source, input_target, label):
        logits_class, logits_domain_source, logits_domain_gru_source, logits_domain_cnn_source = self.model(inputs_source, True)
        _, logits_domain_target, logits_domain_gru_target, logits_domain_cnn_target = self.model(input_target, True)

        label_domain = torch.cat([torch.ones(label.size(0)), torch.zeros(label.size(0))]).long().to(label.device)
        loss_clf = self.criterion(logits_class, label)
        loss_domain = self.criterion(torch.cat([logits_domain_source, logits_domain_target], dim=0), label_domain) \
                    + self.criterion(torch.cat([logits_domain_gru_source, logits_domain_gru_target], dim=0), label_domain) * alpha \
                    + self.criterion(torch.cat([logits_domain_cnn_source, logits_domain_cnn_target], dim=0), label_domain) * beta
        return loss_clf, loss_domain, logits_class
    

    def train(self, train_loader, val_loader, test_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter_train = 0
        logging.info(f"Start SimCLR fine-tuning head for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        acc = 0
        f1 = 0
        best_epoch = 0
        best_acc = 0
        best_f1 = 0

        for epoch_counter in tqdm(range(self.args.epochs)):
            acc_batch = AverageMeter('acc_batch', ':6.2f')
            f1_batch = AverageMeter('f1_batch', ':6.2f')

            for sensor, target in train_loader:
                sensor = sensor.to(self.args.device)
                target = target[:, 0].to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    loss_clf, loss_domain, logits = self.loss_xhar(sensor, sensor, target)
                    loss = loss_clf + loss_domain

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

            val_acc, val_f1 = self.xhar_evaluate(data_loader=val_loader)

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
        return


    def test_performance(self, best_model_dir, test_loader):
        checkpoint = torch.load(best_model_dir, map_location="cpu")
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        test_acc, test_f1 = self.xhar_evaluate(data_loader=test_loader)
        logging.info(f"test f1 is {test_f1}.")
        logging.info(f"test acc is {test_acc}.")
        
        print('test f1 is {} for {}'.format(test_f1, self.args.name))
        print('test acc is {} for {}'.format(test_acc, self.args.name))

    
    def xhar_evaluate(self, data_loader):
        losses = AverageMeter('Loss', ':.4e')
        acc_eval = AverageMeter('acc_eval', ':6.2f')
        f1_eval = AverageMeter('f1_eval', ':6.2f')

        self.model.eval()

        with torch.no_grad():
            for sensor, target in data_loader:
                sensor = sensor.to(self.args.device)
                if sensor.shape == 2:
                    sensor = sensor.unsqueeze(dim=0)
                target = target[:, 0].to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    logits = self.model(sensor, training=False)
                    loss = self.criterion(logits, target)

                losses.update(loss.item(), sensor.size(0))
                _, pred = logits.topk(1, 1, True, True)
                acc = accuracy(logits, target, topk=(1,))
                f1 = f1_cal(logits, target, topk=(1,))
                acc_eval.update(acc, sensor.size(0))
                f1_eval.update(f1, sensor.size(0))

        return acc_eval.avg, f1_eval.avg