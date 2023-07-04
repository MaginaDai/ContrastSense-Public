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
    def __init__(self, temperature, adversarial_weighting_factor=1):
        super(SAContrastiveAdversarialLoss, self).__init__()
        self.BATCH_DIM = 0
        self.tau = temperature
        self.lam = adversarial_weighting_factor
        self.cos_sim = torch.nn.CosineSimilarity(0)
        self.log_noise = 1e-12 # 8 # see https://stackoverflow.com/questions/40050397/deep-learning-nan-loss-reasons
        # self.contrastive_loss = ContrastiveLoss(temperature)
        pass
    
    def forward(self, z1s, z2s, z1_c_outs, z1_subject_labels):
        """
        z1s represents the (batched) representation(s) of the t1-transformed signal(s)
        z2s represents the (batched) representation(s) of the t2-transformed signal(s)
        z1_c_outs represents the (batched) subject predictions produced by the adversary
        z1_subject_labels represents the (batched) subject labels, representing the ground truth for the adversary

        see Sectoin 3.1 of arxiv.org/pdf/2007.04871.pdf
        """
        z1_c_outs = torch.nn.functional.normalize(z1_c_outs, p=2, dim=1) # see https://discuss.pytorch.org/t/how-to-normalize-embedding-vectors/1209

        loss = 0.
        curr_batch_size = z1s.size(self.BATCH_DIM)

        # get contrastive loss of representations
        # loss += self.contrastive_loss(z1s, z2s)
        z1s = z1s.view(curr_batch_size, -1)
        z2s = z2s.view(curr_batch_size, -1)

        for i in range(curr_batch_size):
            # see https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html

            # compute loss contributions of t1-to-other pairings
            numerator1 = torch.exp(self.cos_sim(z1s[i,:], z2s[i,:]) / self.tau)
            denominator1 = 0.
            for k in range(curr_batch_size):
                denominator1 += torch.exp(self.cos_sim(z1s[i,:], z2s[k,:]) / self.tau) # compare t1 ith signal with all t2 signals
                if k != i:                                                             # compare t1 ith signal to all other t1 signals, skipping the t1 ith signal
                    denominator1 += torch.exp(self.cos_sim(z1s[i,:], z1s[k,:]) / self.tau)
            loss += -1.*torch.log(self.log_noise + numerator1/denominator1)
            # print("SAContrastiveAdversarialLoss.forward: \t loss == ", loss)
            
            # SKIP loss contributions of t2-to-other pairings because they came from momentum-updated network
            # numerator2 = torch.exp(self.cos_sim(z2s[i,:], z1s[i,:]) / self.tau)
            # denominator2 = 0.
            # for k in range(curr_batch_size):
            #     denominator2 += torch.exp(self.cos_sim(z2s[i,:], z1s[k,:]) / self.tau) # compare augmented ith signal with all orig signals
            #     if k != i:                                                             # compare augmented ith signal to all other augmented signals, skipping the augmented ith signal
            #         denominator2 += torch.exp(self.cos_sim(z2s[i,:], z2s[k,:]) / self.tau)
            # loss += -1.*torch.log(numerator2/denominator2)

        loss = loss / (curr_batch_size*(2.*curr_batch_size - 1.)) # loss / (curr_batch_size*2.*(2.*curr_batch_size - 1.)) # take the average loss across the t1 signals 

        for i in range(curr_batch_size):
            j = torch.argmax(z1_subject_labels[i,:])
            loss += self.lam *(-1.)*torch.log(self.log_noise + (1. - z1_c_outs[i,j])) # see equation 3 of arxiv.org/pdf/2007.04871.pdf
            # print("SAContrastiveAdversarialLoss.forward: \t loss == ", loss)
        
        return loss

    def get_number_of_correct_reps(self, z1s, z2s, z1_c_outs, z1_subject_labels):
        curr_batch_size = z1s.size(self.BATCH_DIM)

        z1s = z1s.view(curr_batch_size, -1)
        z2s = z2s.view(curr_batch_size, -1)

        num_correct_reps = 0.
        for i in range(curr_batch_size):
            # see https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html

            # compute accuracy contributions of orig-to-other pairings
            sim_measure_of_interest = self.cos_sim(z1s[i,:], z2s[i,:])
            representation_is_correct = True
            for k in range(curr_batch_size):
                other_sim_measure = self.cos_sim(z1s[i,:], z2s[k,:]) # compare t1 ith signal with all augmented signals
                if other_sim_measure > sim_measure_of_interest:
                    representation_is_correct = False
                    break
                if k != i:                                           # compare t1 ith signal to all other orig signals, skipping the t1 ith signal
                    other_sim_measure = self.cos_sim(z1s[i,:], z1s[k,:])
                    if other_sim_measure > sim_measure_of_interest:
                        representation_is_correct = False
                        break
                
            if torch.argmax(z1_subject_labels[i,:]) == torch.argmax(z1_c_outs[i,:]):
                representation_is_correct = False

            if representation_is_correct:
                num_correct_reps += 1.
            
            # SKIP loss contributions of t2-to-other pairings because they were generated by momentum-updated network
            # sim_measure_of_interest = self.cos_sim(z2s[i,:], z1s[i,:])
            # representation_is_correct = True
            # for k in range(curr_batch_size):
            #     other_sim_measure += self.cos_sim(z2s[i,:], z1s[k,:]) # compare augmented ith signal with all orig signals
            #     if other_sim_measure > sim_measure_of_interest:
            #         representation_is_correct = False
            #         break
            #     if k != i:                                                             # compare augmented ith signal to all other augmented signals, skipping the augmented ith signal
            #         other_sim_measure += self.cos_sim(z2s[i,:], z2s[k,:])
            #         if other_sim_measure > sim_measure_of_interest:
            #             representation_is_correct = False
            #             break
            # if representation_is_correct:
            #     num_correct_reps += 1.

        return num_correct_reps

class SAAdversarialLoss(nn.Module):
    """
    see Section 3.1 of arxiv.org/pdf/2007.04871.pdf
    """
    def __init__(self):
        super(SAAdversarialLoss, self).__init__()
        self.BATCH_DIM = 0
        self.log_noise = 1e-12 # 8 # see https://stackoverflow.com/questions/40050397/deep-learning-nan-loss-reasons
        pass
    
    def forward(self, z1_c_outs, z1_subject_labels):
        """
        z1_c_outs represents the (batched) subject predictions produced by the adversary
        z1_subject_labels represents the (batched) subject labels, representing the ground truth for the adversary

        see Sectoin 3.1 of arxiv.org/pdf/2007.04871.pdf
        """
        # print("z1_c_outs.shape == ", z1_c_outs.shape)
        # print("z1_c_outs == ", z1_c_outs)
        z1_c_outs = torch.nn.functional.normalize(z1_c_outs, p=2, dim=1) # see https://discuss.pytorch.org/t/how-to-normalize-embedding-vectors/1209
        # print("z1_c_outs == ", z1_c_outs)
        # print("z1_c_outs.shape == ", z1_c_outs.shape)

        loss = 0.
        curr_batch_size = z1_c_outs.size(self.BATCH_DIM)

        for i in range(curr_batch_size):
            j = torch.argmax(z1_subject_labels[i,:])
            loss += -1.*torch.log(self.log_noise + z1_c_outs[i,j]) # see equation 3 of arxiv.org/pdf/2007.04871.pdf
            # print("SAAdversarialLoss.forward: \t loss == ", loss, " (i,j) == ", (i,j), " z1_c_outs[i,j] == ", z1_c_outs[i,j])
        # raise NotImplementedError()
        return loss

    def get_number_of_correct_preds(self, z1_c_outs, z1_subject_labels):
        num_correct_preds = 0.
        curr_batch_size = z1_c_outs.size(self.BATCH_DIM)
        
        for i in range(curr_batch_size):
            if torch.argmax(z1_subject_labels[i,:]) == torch.argmax(z1_c_outs[i,:]):
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
        self.adversarial_optimizer = kwargs['adversarial_optimizer']

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

        self.loss_fn = SAContrastiveAdversarialLoss(temperature=0.05, adversarial_weighting_factor=1.0)
        self.adversarial_loss_fn = SAAdversarialLoss()


    
    def train(self, train_loader, val_loader):
        scaler_ad = GradScaler(enabled=self.args.fp16_precision)
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
            self.model.train()

            acc_batch = AverageMeter('acc_batch', ':6.2f')
            loss_batch = AverageMeter('loss_batch', ':6.5f')
            running_train_loss = 0
            num_correct_train_preds = 0
            total_num_train_preds = 0
            running_adversary_train_loss = 0
            num_adversary_correct_train_preds = 0
            total_num_adversary_train_preds = 0

            for sensor, domain_label in train_loader:
                x1 = sensor[:, 0].to(self.args.device)
                x2 = sensor[:, 1].to(self.args.device)
                domain_label = domain_label.to(self.args.device)

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
                
                num_adversary_correct_train_preds += self.adversarial_loss_fn.get_number_of_correct_preds(features, domain_label)
                total_num_adversary_train_preds += len(features)

                scaler_ad.scale(adversarial_loss).backward()
                scaler_ad.step(self.adversarial_optimizer)
                scaler_ad.update()
                running_adversary_train_loss += adversarial_loss.item()

                ## training for high-level features

                for p in self.model.model.parameters():
                    p.requires_grad = True
                for p in self.model.momentum_model.parameters():
                    p.requires_grad = False
                for p in self.model.adversary.parameters():
                    p.requires_grad = False

                self.optimizer.zero_grad()

                with autocast(enabled=self.args.fp16_precision):
                    x1_rep, x2_rep, x1_subject_preds = self.model(x1, x2)
                    loss = self.loss_fn(x1_rep, x2_rep, x1_subject_preds, domain_label)
                
                num_correct_train_preds += self.loss_fn.get_number_of_correct_reps(x1_rep, x2_rep, x1_subject_preds, domain_label)
                total_num_train_preds += len(x1_rep)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                running_train_loss += loss.item()
                self.model.momentum_model = momentum_model_parameter_update(self.args.momentum, self.model.momentum_model, self.model.model)

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc', acc, global_step=n_iter)
                    self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], global_step=n_iter)
                n_iter += 1
            
            val_acc, val_loss = self.pretrain_evaluate(model=self.model, data_loader=val_loader)
            self.writer.add_scalar('loss_acc', val_acc, global_step=epoch_counter)
            self.writer.add_scalar('loss_eval', val_loss, global_step=epoch_counter)

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