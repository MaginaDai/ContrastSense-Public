from cProfile import label
from contextlib import AsyncExitStack
import logging
import pdb
from re import T
import sys
from os.path import dirname

sys.path.append(dirname(dirname(sys.path[0])))
from tkinter.ttk import LabeledScale
from tqdm import tqdm
from judge import AverageMeter
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from kmeans_pytorch import kmeans
from utils import accuracy, evaluate, f1_cal, save_checkpoint, save_config_file

N_VIEWS = 2
LARGE_NUM = 1e9

class  SimCLR_cluster(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']

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

    def info_nce_loss(self, features):  # something to improve.

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(N_VIEWS)], dim=0)
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
    
    def cluster_loss(self, p1, p2):
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)
        labels = torch.arange(0, p1.shape[0])

        logits1 = self.sim(p1, p2, self.args.temperature)
        logits2 = self.sim(p2, p1, self.args.temperature)
        return logits1, logits2, labels

    def sim(self, p1, p2, t):
        logits_aa = torch.matmul(p1, p1.T) / t
        logits_ab = torch.matmul(p1, p2.T) / t

        p1_clu, _ = kmeans(X=p1, num_clusters=6, device=self.args.device, if_tqdm=False)
        # print(p1_clu)
        masks_aa = (p1_clu.unsqueeze(0) == p1_clu.unsqueeze(1)).float()
        logits_aa = logits_aa - masks_aa * LARGE_NUM

        masks_ab = masks_aa - torch.eye(p1.shape[0]).to(self.args.device)
        logits_ab = logits_ab - masks_ab * LARGE_NUM

        return torch.concat([logits_ab, logits_aa], axis=1)

    def cluster_loss_optimized(self, features):
        features = F.normalize(features, dim=1)
        num = int(features.shape[0]/2)
        p = features[:num, :].detach().clone()
        p_clu, self.cluster_centers = kmeans(X=p, num_clusters=6, device=self.args.device, if_tqdm=False)

        masks_aa = (p_clu.unsqueeze(0) == p_clu.unsqueeze(1)).float()
        masks_ab = masks_aa - torch.eye(p.shape[0]).to(self.args.device)

        masks_1 = torch.concat([masks_aa, masks_ab], axis=1)
        masks_2 = torch.concat([masks_ab, masks_aa], axis=1)

        masks = torch.concat([masks_1, masks_2], axis=0)
        
        logits = torch.matmul(features, features.T)

        logits = logits - masks * LARGE_NUM
        
        labels = torch.concat([torch.arange(num, 2*num), torch.arange(0, num)], axis=0).to(self.args.device)
        return logits, labels


    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start ClusterCLHAR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {~self.args.disable_cuda}.")

        acc = 0
        best_epoch = 0
        best_acc = 0

        for epoch_counter in tqdm(range(self.args.epochs)):
            acc_batch = AverageMeter('acc_batch', ':6.2f')
            loss_batch = AverageMeter('loss_batch', ':6.5f')
            for sensor, _ in train_loader:
                # sensor = torch.cat(sensor, dim=0)
                # sensor = sensor.to(self.args.device)
                sensor[0] = sensor[0].to(self.args.device)
                sensor[1] = sensor[1].to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    ##### Improved
                    # features = self.model(sensor)
                    # logits, labels = self.cluster_loss_optimized(features)
                    # loss = self.criterion(logits, labels)

                    ##### original
                    p1 = self.model(sensor[0])
                    p2 = self.model(sensor[1])
                    
                    if self.args.do_cluster:
                        logits1, logits2, labels = self.cluster_loss(p1, p2)
                        labels = labels.to(self.args.device)
                        loss1 = self.criterion(logits1, labels)
                        loss2 = self.criterion(logits2, labels)

                        loss = loss1 + loss2
                    else:
                        features = torch.concat([p1, p2], dim=0)
                        logits, labels = self.info_nce_loss(features)
                        loss = self.criterion(logits, labels)
                        loss1 = 0
                        loss2 = 0
                        


                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                
                if self.args.do_cluster:
                    acc = 0.5 * (accuracy(logits1, labels, topk=(1,)) + accuracy(logits2, labels, topk=(1,)))
                else:
                    acc = accuracy(logits, labels, topk=(1,))
                acc_batch.update(acc, sensor[0].size(0))
                loss_batch.update(loss, sensor[0].size(0))

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc', acc, global_step=n_iter)
                    # self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], global_step=n_iter)

                n_iter += 1

            is_best = acc_batch.avg >= best_acc
            best_acc = max(acc_batch.avg, best_acc)
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
        logging.info(f"Start SimCLR fine-tuning head for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        if not self.args.if_fine_tune:
            """
            Switch to eval mode:
            Under the protocol of linear classification on frozen features/models,
            it is not legitimate to change any part of the pre-trained model.
            BatchNorm in train mode may revise running mean/std (even if it receives
            no gradient), which are part of the model parameters too.
            """
            self.model.eval()

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
                    # self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], global_step=n_iter_train)

                n_iter_train += 1

            val_acc, val_f1 = evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=val_loader)
            if self.args.if_fine_tune:
                self.model.train()

            is_best = val_f1 > best_f1
            best_f1 = max(val_f1, best_f1)
            best_acc = max(val_acc, best_acc)
            if is_best:
                best_epoch = epoch_counter
                checkpoint_name = 'model_best.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter + 1,
                    'state_dict': self.model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, filename=os.path.join(self.writer.log_dir, checkpoint_name), path_dir=self.writer.log_dir)

            self.writer.add_scalar('eval acc', val_acc, global_step=epoch_counter)
            self.writer.add_scalar('eval f1', val_f1, global_step=epoch_counter)
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

