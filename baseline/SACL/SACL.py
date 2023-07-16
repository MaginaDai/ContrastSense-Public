import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
import os
import sys
import numpy as np
from os.path import dirname

sys.path.append(dirname(dirname(sys.path[0])))


from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from judge import AverageMeter
from utils import f1_cal, save_config_file, accuracy, save_checkpoint, evaluate, split_last, merge_last, CPC_evaluate
from sklearn.metrics import f1_score

### code from https://github.com/zacharycbrown/ssl_baselines_for_biosignal_feature_extraction

class SAContrastiveAdversarialLoss(nn.Module):
    """
    see Section 3.1 of arxiv.org/pdf/2007.04871.pdf
    """
    def __init__(self, device, temperature, adversarial_weighting_factor=1):
        super(SAContrastiveAdversarialLoss, self).__init__()
        self.BATCH_DIM = 0
        self.device = device
        self.tau = temperature
        self.lam = adversarial_weighting_factor
        self.cos_sim = torch.nn.CosineSimilarity(0)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.log_noise = 1e-12 # 8 # see https://stackoverflow.com/questions/40050397/deep-learning-nan-loss-reasons
        # self.contrastive_loss = ContrastiveLoss(temperature)
        pass
    
    def forward(self, z1s, z2s, z1_c_outs, z1_subject_labels):
        loss = 0.
        curr_batch_size = z1s.size(self.BATCH_DIM)

        z1s = z1s.view(curr_batch_size, -1)
        z2s = z2s.view(curr_batch_size, -1)

        features = torch.cat([z1s, z2s], axis=0)

        labels = torch.cat([torch.arange(curr_batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)
        z1_c_outs = F.normalize(z1_c_outs, p=2, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        logits = logits / self.tau

        acc = accuracy(logits, labels, topk=(1,))
        loss = self.criterion(logits, labels)

        for i in range(curr_batch_size):
            j = z1_subject_labels[i]
            loss += self.lam *(-1.)*torch.log(self.log_noise + (1. - z1_c_outs[i,j])) / curr_batch_size # see equation 3 of arxiv.org/pdf/2007.04871.pdf

        return loss, acc

class SAAdversarialLoss(nn.Module):
    """
    see Section 3.1 of arxiv.org/pdf/2007.04871.pdf
    """
    def __init__(self, device):
        super(SAAdversarialLoss, self).__init__()
        self.BATCH_DIM = 0
        self.log_noise = 1e-12 # 8 # see https://stackoverflow.com/questions/40050397/deep-learning-nan-loss-reasons
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        pass
    
    def forward(self, logits, domain_label):
        loss = self.criterion(logits, domain_label)
        return loss

    def get_number_of_correct_preds(self, z1_c_outs, z1_subject_labels):
        num_correct_preds = 0.
        curr_batch_size = z1_c_outs.size(self.BATCH_DIM)
        
        for i in range(curr_batch_size):
            if z1_subject_labels[i]== torch.argmax(z1_c_outs[i,:]):
                num_correct_preds += 1.

        return num_correct_preds

def momentum_model_parameter_update(momentum_factor, momentum_model, orig_model): # see https://discuss.pytorch.org/t/copying-weights-from-one-net-to-another/1492/16
    for momentum_model_param, orig_model_param in zip(momentum_model.parameters(), orig_model.parameters()):
        momentum_model_param.copy_(momentum_factor*momentum_model_param.data + (1.-momentum_factor)*orig_model_param.data)
    return momentum_model


class SACL(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        if not self.args.transfer:
            self.adversarial_optimizer = kwargs['adversarial_optimizer']

        writer_pos = './runs/' + self.args.store + '/' + self.args.name
        if self.args.transfer is True:
            writer_pos += '_ft'
            writer_pos += f'_shot_{self.args.shot}'
        else:
            writer_pos += '/'
        self.writer = SummaryWriter(writer_pos)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

        self.loss_fn = SAContrastiveAdversarialLoss(device=self.args.device, temperature=0.05, adversarial_weighting_factor=1.0)
        self.adversarial_loss_fn = SAAdversarialLoss(self.args.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def train(self, train_loader, val_loader):
        scaler_ad = GradScaler(enabled=self.args.fp16_precision)
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SACL training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        acc = 0
        loss = 0
        best_epoch = 0
        best_loss = 1e9

        for epoch_counter in tqdm(range(self.args.epochs)):
            self.model.train()

            acc_batch = AverageMeter('acc_batch', ':6.2f')
            ad_acc_batch = AverageMeter('ad_acc_batch', ':6.2f')
            loss_batch = AverageMeter('loss_batch', ':6.5f')
            ad_loss_batch = AverageMeter('ad_loss_batch', ':6.5f')

            for sensor, label in train_loader:
                x1 = sensor[0].to(self.args.device)
                x2 = sensor[1].to(self.args.device)
                domain_label = label[:, 1].to(self.args.device)

                ## adversarial training for domain-invariant features
                for p in self.model.model.parameters():
                    p.requires_grad = False
                for p in self.model.momentum_model.parameters():
                    p.requires_grad = False
                for p in self.model.adversary.parameters():
                    p.requires_grad = True
                
                self.adversarial_optimizer.zero_grad()
                
                with autocast(enabled=self.args.fp16_precision):
                    features = self.model.adversarial_forward(x1)
                    adversarial_loss = self.adversarial_loss_fn(features, domain_label)

                scaler_ad.scale(adversarial_loss).backward()
                scaler_ad.step(self.adversarial_optimizer)
                scaler_ad.update()

                ad_acc = accuracy(features, domain_label, topk=(1, ))
                ad_acc_batch.update(ad_acc, x1.size(0))
                ad_loss_batch.update(adversarial_loss, x1.size(0))

                ## training for high-level features

                for p in self.model.model.parameters():
                    p.requires_grad = True
                for p in self.model.momentum_model.parameters():
                    p.requires_grad = False
                for p in self.model.adversary.parameters():
                    p.requires_grad = False

                self.optimizer.zero_grad()

                with autocast(enabled=self.args.fp16_precision):
                    x1_rep, x2_rep, x1_embeds, x1_subject_preds = self.model(x1, x2)
                    loss, acc = self.loss_fn(x1_rep, x2_rep, x1_subject_preds, domain_label)
                
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                acc_batch.update(acc, x1.size(0))
                loss_batch.update(loss, x1.size(0))

                self.model.momentum_model = momentum_model_parameter_update(0.999, self.model.momentum_model, self.model.model)

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('ad_loss', adversarial_loss, global_step=n_iter)
                    self.writer.add_scalar('acc', acc, global_step=n_iter)
                    self.writer.add_scalar('ad_acc', ad_acc, global_step=n_iter)
                    # self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], global_step=n_iter)
                n_iter += 1

            is_best = loss_batch.avg < best_loss
            best_loss = min(loss_batch.avg, best_loss)
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
            logging.debug(f"Epoch: {epoch_counter} loss: {loss_batch.avg} acc: {acc_batch.avg} ad_loss: {ad_loss_batch.avg} ad_acc: {ad_acc_batch.avg}")

            if acc_batch.avg > 99:
                break ## contrastive learning is good enough


        logging.info("Training has finished.")
        logging.info(f"Model of Epoch {best_epoch} checkpoint and metadata has been saved at {self.writer.log_dir}.")

        return
    
    def evaluate(self, val_loader):
        num_correct_val_preds = 0
        total_num_val_preds = 0
        num_correct_adversarial_val_preds = 0
        total_num_adversarial_val_preds = 0

        self.model.eval()
        for sensor, domain_label in val_loader:
            x_t1 = sensor[:, 0]
            x_t2 = sensor[:, 1]
            x_t1, x_t2, domain_label = x_t1.to(self.args.device), x_t2.to(self.args.device), domain_label.to(self.args.device)

            # evaluate model and adversary
            x1_rep = self.model.model(x_t1)
            x2_rep = self.model.momentum_model(x_t2)
            x1_embeds = self.model.embed_model(x_t1)
            x1_subject_preds = self.model.adversary(x1_embeds)
            # x1_subject_preds = adversary(x1_rep)

            num_correct_val_preds += self.loss_fn.get_number_of_correct_reps(x1_rep, x2_rep, x1_subject_preds, domain_label)
            total_num_val_preds += len(x1_rep)

            num_correct_adversarial_val_preds += self.adversarial_loss_fn.get_number_of_correct_preds(x1_subject_preds, domain_label)
            total_num_adversarial_val_preds += len(x1_subject_preds)

        return
    
    def transfer_train(self, tune_loader, val_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter_train = 0
        logging.info(f"Start SACL fine-tuning for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        acc = 0
        f1 = 0
        best_epoch = 0
        best_acc = 0
        best_f1 = 0
        not_best_counter = 0
        for epoch_counter in tqdm(range(self.args.epochs)):
            acc_batch = AverageMeter('acc_batch', ':6.2f')

            pred_batch = torch.empty(0).to(self.args.device)
            label_batch = torch.empty(0).to(self.args.device)
            self.model.train()
            
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

                n_iter_train += 1

            f1_batch = f1_score(label_batch.cpu().numpy(), pred_batch.cpu().numpy(), average='macro') * 100
            val_acc, val_f1 = evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=val_loader)

            is_best = val_f1 > best_f1
            best_f1 = max(val_f1, best_f1)
            best_acc = max(val_acc, best_acc)
            if is_best:
                not_best_counter = 0
                best_epoch = epoch_counter
                checkpoint_name = 'model_best.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter,
                    'state_dict': self.model.state_dict(),
                    'best_f1': best_f1,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, filename=os.path.join(self.writer.log_dir, checkpoint_name), path_dir=self.writer.log_dir)
            else:
                not_best_counter += 1

            if not_best_counter >= 30: # early stop
                break

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