import logging
import sys
from os.path import dirname
from sklearn.metrics import f1_score

from MoCo import GradientReversal
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


class CALDA_encoder(nn.Module):
    def __init__(self, num_classes, num_domains, num_units, modal):

        super().__init__()
        self.num_classes = num_classes
        self.num_domains = num_domains

        self.dropout = torch.nn.Dropout(p=0.3)
        self.relu = torch.nn.ReLU()

        self.feature_extractor = nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=(8, 1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=(5, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 128, kernel_size=(3, 1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            )

        if modal == 'imu':
            self.feature_size = 1122
        elif modal == 'emg':
            self.feature_size = 312
        else:
            NotADirectoryError

        self.task_classifier = torch.nn.Linear(self.feature_size, num_classes)

        self.domain_classifier = nn.Sequential(
            GradientReversal(),
            torch.nn.Linear(self.feature_size, 500),
            torch.nn.BatchNorm1d(500),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(500, 500),
            torch.nn.BatchNorm1d(500),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(500, num_domains),
        )

        self.contrastive_head = torch.nn.Linear(self.feature_size, num_units)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        h = self.feature_extractor(x)
        h = h.permute([0, 2, 1, 3])
        h = F.max_pool2d(h, kernel_size=(h.size()[2], 1)) 
        h = h.reshape(h.shape[0], -1)
        y_logits = self.task_classifier(h)
        domain_logits = self.domain_classifier(h)
        contrast_logits = self.contrastive_head(h)
        return y_logits, domain_logits, contrast_logits

class CALDA():
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.device = self.args.device

        writer_pos = './runs/' + self.args.store + '/' + self.args.name
        if self.args.shot >=0:
            writer_pos += f'_shot_{self.args.shot}'
        else:
            writer_pos += f'_percent_{self.args.percent}'
        self.writer = SummaryWriter(writer_pos)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        self.weight_adversary = self.args.weight_adversary
        self.weight_similarity = self.args.weight_similarity
    
    def compute_losses(self, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, contrastive_output):
        
        # Normal task and domain classifier losses
        task_loss = self.criterion(task_y_pred, task_y_true)
        d_loss = self.criterion(domain_y_pred, domain_y_true)

        # Additional contrastive loss
        similarity_loss = self._contrastive_loss(task_y_true, contrastive_output)

        # If weak supervision, include it in the total loss and also return it
        # for plotting in metrics
        total_loss = task_loss \
            + self.weight_adversary*d_loss \
            + self.weight_similarity*similarity_loss
        return [total_loss, task_loss, d_loss, similarity_loss]
    
    def _contrastive_loss(self, task_y_true, z_output):
        z_output = F.normalize(z_output, dim=1)
        task_y_true = task_y_true.unsqueeze(1)

        similarity = torch.matmul(z_output, z_output.T)
        similarity = similarity / self.args.temperature

        diag_mask = torch.eye(z_output.shape[0], dtype=torch.bool).to(self.device)
        mask = torch.eq(task_y_true, task_y_true.T).float() - diag_mask.float()
        
        mask = mask.float().to(self.device)
        
        exp_logits = torch.exp(similarity)
        exp_logits = exp_logits[~diag_mask].view(z_output.shape[0], -1)
        log_prob = similarity - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos.mean()
        return loss
    
    def train(self, train_loader, val_loader):
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

            self.model.train()

            for sensor, target in train_loader:
                sensor = sensor.to(self.args.device)
                y_target = target[:, 0].to(self.args.device)
                domain_target = target[:, 1].to(self.args.device)

                y_logits, domain_logits, contrast_logits = self.model(sensor)

                loss, _, _, _ = self.compute_losses(y_target, domain_target, y_logits, domain_logits, contrast_logits)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                acc = accuracy(y_logits, y_target, topk=(1,))
                f1 = f1_cal(y_logits, y_target, topk=(1,))
                acc_batch.update(acc, sensor.size(0))

                label_batch = torch.cat((label_batch, y_target))
                _, pred = y_logits.topk(1, 1, True, True)
                pred_batch = torch.cat((pred_batch, pred.reshape(-1)))

                if n_iter_train % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter_train)
                    self.writer.add_scalar('acc', acc, global_step=n_iter_train)
                    self.writer.add_scalar('f1', f1, global_step=n_iter_train)

                n_iter_train += 1
            
            f1_batch = f1_score(label_batch.cpu().numpy(), pred_batch.cpu().numpy(), average='macro') * 100
            val_acc, val_f1 = evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=val_loader)

            is_best = val_acc > best_acc  # follow the original selection criteria
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

            self.writer.add_scalar('eval acc', val_acc, global_step=epoch_counter)
            self.writer.add_scalar('eval f1', val_f1, global_step=epoch_counter)
            logging.debug(f"Epoch: {epoch_counter} Loss: {loss} acc: {acc_batch.avg: .3f}/{val_acc: .3f} f1: {f1_batch: .3f}/{val_f1: .3f}")

        logging.info("Training has finished.")
        logging.info(f"best eval acc is {best_acc} at {best_epoch}.")

        print('best eval acc is {} for {}'.format(best_acc, self.args.name))

        return

    def test_performance(self, best_model_dir, test_loader):
        checkpoint = torch.load(best_model_dir, map_location="cpu")
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        test_acc, test_f1 = evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=test_loader)
        logging.info(f"test f1 is {test_f1}.")
        logging.info(f"test acc is {test_acc}.")
        
        print('test f1 is {} for {}'.format(test_f1, self.args.name))
        print('test acc is {} for {}'.format(test_acc, self.args.name))
        return