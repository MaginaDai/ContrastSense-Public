import os, time
import logging
import pdb
from re import I
from tkinter import Y
from tkinter.messagebox import NO
from unicodedata import bidirectional
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from DeepSense import DeepSense_encoder
from MoCo import MoCo_model
from baseline.SACL.model import SACLEncoder_CNN_block

from SupContrast import SupConLoss
from baseline.CDA.model import TCN_GCN_unit
from data_aug.preprocessing import UsersNum
from figure_plot.figure_plot import t_SNE_view
from judge import AverageMeter
from neg_select import PCA_torch

from simclr import MultiHeadedSelfAttention, LayerNorm, PositionWiseFeedForward
from utils import f1_cal, save_config_file, save_checkpoint, accuracy, CPC_evaluate, MoCo_evaluate
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, adjusted_rand_score
from sklearn.manifold import TSNE
# from kmeans_pytorch import kmeans
from sklearn.metrics import f1_score

class ContrastSense_without_queue(nn.Module):
    def __init__(self, args, transfer=False, classes=6, out_dim=256, users_class=None):
        super(ContrastSense_without_queue, self).__init__()

        device=args.device
        modal=args.modal
        modal = args.modal

        self.K = args.moco_K 
        self.m = args.moco_m
        self.T = args.temperature
        self.T_labels = args.tem_labels
        self.label_type = args.label_type

        self.if_cross_entropy = args.CE
        self.DAL = args.DAL

        self.modal = args.modal

        self.batch_size = args.batch_size
        self.hard_sample = args.hard
        self.sample_ratio = args.sample_ratio
        self.last_ratio = args.last_ratio
        self.hard_record = args.hard_record
        self.time_window = args.time_window
        self.scale_ratio = args.scale_ratio
        self.device = device
        

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = MoCo_model(transfer=transfer, classes=classes, DAL=self.DAL, users_class=users_class, modal=modal)
        self.encoder_k = MoCo_model(transfer=transfer, DAL=self.DAL, users_class=users_class, modal=modal)


        self.sup_loss = SupConLoss(device=device, if_cross_entropy=self.if_cross_entropy)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        self.tsne = TSNE(n_components=2, learning_rate='auto', init='random')

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, sen_q, sen_k, domain_label, time_label):
        """
        Input:
            sen_q: a batch of query sensors data
            sen_k: a batch of key sensors data
            domain_label: the label for nuisance suppression
        Output:
            logits, targets
        """

        device = sen_q.device

        hardest_related_info = [0, 0, 0, 0, 0]
        similarity_across_domains = 0

        q, _ = self.encoder_q(sen_q)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k, _ = self.encoder_k(sen_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, k.T]) 
        
        if self.hard_sample:
            #### v10 domain-wise sorting + time window
            if self.last_ratio < 1.0:  ## if last_ratio >= 1, then we dont apple this simplist elimination. 
                start_time = time.time()
                domains_in_queues = torch.unique(domain_label[0]).contiguous().view(-1, 1)
                domain_queues_mask = torch.eq(domains_in_queues, domain_label[0].reshape(1, -1)).bool().to(device)
                neg_for_sampling = l_neg.clone().detach()

                for j, domain_for_compare in enumerate(domains_in_queues):
                    key_in_domain_j = domain_queues_mask[j].repeat(neg_for_sampling.shape[0], 1)
                    domain_queue_j = neg_for_sampling[key_in_domain_j].view(neg_for_sampling.shape[0], -1)
                    _, indices = torch.sort(domain_queue_j, dim=1, descending=True)
                    idx_to_eliminate = indices[:, int(domain_queue_j.shape[1] * self.last_ratio):]
                    position = torch.where(domain_queues_mask[j] == True)[0].repeat(neg_for_sampling.shape[0], 1)

                    rows = torch.arange(neg_for_sampling.shape[0]).unsqueeze(-1)
                    masks_for_domain_j = torch.zeros(neg_for_sampling.shape).bool().to(device)
                    masks_for_domain_j[rows, position[rows, idx_to_eliminate]] = True
                    l_neg[masks_for_domain_j] = -torch.inf
                    hardest_related_info[1] += idx_to_eliminate.shape[1]
                
                hardest_related_info[1]  /= len(domains_in_queues)  ## record the avg number of eliminated samples
            
                end_time = time.time()
                hardest_related_info[2] = end_time - start_time

            if self.time_window != 0:
                start_time = time.time()
                low_boundary = time_label[0].contiguous().view(-1, 1) - self.time_window
                high_boundary = time_label[0].contiguous().view(-1, 1) + self.time_window
                queue_time = time_label[0].expand(l_neg.shape[0], l_neg.shape[1])
                mask_low = low_boundary < queue_time # (NxQ) = label of sen_q (Nx1) x labels of queue (Qx1).T
                mask_high = queue_time < high_boundary
                mask = torch.logical_and(mask_low, mask_high)
                l_neg[mask] = -torch.inf    
                hardest_related_info[0] = mask.sum(1).float().mean()
                
                end_time = time.time()
                hardest_related_info[3] = end_time - start_time


        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # targets: positive key indicators
        targets = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        feature = torch.einsum('nc,ck->nk', [q, k.T])
        
        return logits, targets, feature, hardest_related_info, similarity_across_domains
    

    def supervised_CL(self, logits_labels=None, labels=None):
        if labels and self.label_type:
            loss = torch.zeros(self.label_type)
            for i in range(self.label_type):
                 # since we dont have queue, now the domain labels of negatives are the same as the positives
                loss[i] = self.sup_loss(logits_labels/self.T_labels[i], labels=labels[i], queue_labels=labels[i].reshape(-1, 1))
            return loss
        else:
            return None
    

class MoCo_without_queue(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        if self.args.DAL and self.args.transfer:
            self.optimizer_DAL = kwargs['optimizer_DAL']
            self.scheduler_DAL = kwargs['scheduler_DAL']
        
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

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start MoCo training without queue for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")

        acc = 0
        best_epoch = 0
        best_acc = 0
        not_best_counter = 0
        best_loss = 1e6

        if self.args.time_analysis:
            time_tw = AverageMeter('time_tw_avg', ':6.5f')
            time_ss = AverageMeter('time_ss_avg', ':6.5f')
            time_ef = AverageMeter('time_ef_avg', ':6.5f')
            time_fw = AverageMeter('time_fw_avg', ':6.5f')

        for epoch_counter in tqdm(range(self.args.epochs)):
            acc_batch = AverageMeter('acc_batch', ':6.2f')
            loss_batch = AverageMeter('loss_batch', ':6.5f')
            mscr_batch = AverageMeter('mslr_batch', ':6.3f')
            msdr_batch = AverageMeter('msdr_batch', ':6.3f')
            mscsdr_batch = AverageMeter('mscsdr_batch', ':6.3f')

            start_time = time.time()
            for sensor, labels in train_loader:
                
                sensor = [t.to(self.args.device) for t in sensor]
                class_label = labels[:, 0].to(self.args.device) # the first dim is motion labels

                if self.args.label_type or self.args.hard:
                    time_label = [labels[:, -1].to(self.args.device)] # the last dim is time labels
                    if self.args.cross == 'users': # use domain labels
                        domain_label = [labels[:, 1].to(self.args.device)] 
                    elif self.args.cross == 'positions' or self.args.cross == 'devices' :
                        domain_label = [labels[:, 2].to(self.args.device)] 
                    elif self.args.cross == 'multiple':
                        domain_label = [labels[:, 3].to(self.args.device)]
                    else:
                        NotADirectoryError
                else:
                    domain_label = None
                    time_label = None
                with autocast(enabled=self.args.fp16_precision):
                    output, target, logits_domain, hardest_related_info, similarity_across_domains = self.model(sensor[0], sensor[1], 
                                                                                                                domain_label=domain_label,
                                                                                                                time_label=time_label)

                    sup_loss = self.model.supervised_CL(logits_labels=logits_domain, labels=domain_label)

                    loss = self.criterion(output, target)
                    ori_loss = loss.detach().clone()

                    if sup_loss is not None:
                        for i in range(len(sup_loss)):
                            loss -= self.args.slr[i] * sup_loss[i]

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                end_time = time.time()

                acc = accuracy(output, target, topk=(1,))

                acc_batch.update(acc, sensor[0].size(0))
                loss_batch.update(loss, sensor[0].size(0))

                mscr_batch.update(hardest_related_info[0], sensor[0].size(0))
                msdr_batch.update(hardest_related_info[1], sensor[0].size(0))
                mscsdr_batch.update(hardest_related_info[2], sensor[0].size(0))
                
                if n_iter % self.args.log_every_n_steps == 0 and n_iter != 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc', acc, global_step=n_iter)
                    self.writer.add_scalar('mslr', hardest_related_info[0], global_step=n_iter)
                    self.writer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], global_step=n_iter)
                    if sup_loss is not None:
                        self.writer.add_scalar('ori_loss_{}'.format(i), ori_loss, global_step=n_iter)
                        for i in range(len(sup_loss)):
                            self.writer.add_scalar('sup_loss_{}'.format(i), sup_loss[i], global_step=n_iter)

                n_iter += 1

                if self.args.time_analysis:
                    time_sim_select = hardest_related_info[2]
                    time_time_window = hardest_related_info[3]
                    time_encode = hardest_related_info[4]
                    time_full = end_time - start_time

                    time_tw.update(time_time_window, 1)
                    time_ss.update(time_sim_select, 1)
                    time_ef.update(time_encode, 1)
                    time_fw.update(time_full, 1)
                    

                    if n_iter > 5000:
                        break
            
            is_best = loss_batch.avg <= best_loss
            if epoch_counter >= 10:  # only after the first 10 epochs, the best_acc is updated.
                best_loss = min(loss_batch.avg, best_loss)
                best_acc = max(acc_batch.avg, best_acc)
            if is_best:
                best_epoch = epoch_counter
                checkpoint_name = 'model_best.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter,
                    'state_dict': self.model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, filename=os.path.join(self.writer.log_dir, checkpoint_name), path_dir=self.writer.log_dir)
                not_best_counter = 0
            else:
                not_best_counter += 1
            

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            
            if epoch_counter == 0:
                continue  # the first epoch would not have record.
            log_str = f"Epoch: {epoch_counter} Loss: {loss_batch.avg} accuracy: {acc_batch.avg} "
            if self.args.hard:
                log_str += f"sim: {similarity_across_domains} mscr: {mscr_batch.avg} nes: {msdr_batch.avg}"
            if self.args.hard_record:
                log_str += f"mscr: {mscr_batch.avg} msdr: {msdr_batch.avg} mscsdr: {mscsdr_batch.avg}"
            if sup_loss is not None:
                log_str += f"ori_Loss :{ori_loss} "
                for i in range(len(sup_loss)):
                    log_str += f"sup_loss_{i}: {sup_loss[i]} "
            # if cluster_eval is not None:
            #     log_str += f"chs: {chs.avg} center_shift: {center_shift}"
            logging.debug(log_str)

            # if best_acc > 99 and epoch_counter >= 50:
            #     print(f"early stop at {epoch_counter}")
            #     break  # early stop

            if not_best_counter >= 200:
                print(f"early stop at {epoch_counter}")
                break
            

            if n_iter > 5000 and self.args.time_analysis:
                print(f"time for time window selection: {time_tw.avg}")
                print(f"time for similarity selection: {time_ss.avg}")
                print(f"time for feature encoding: {time_ef.avg}")
                print(f"time for forward process: {time_fw.avg}")

                logging.info(f"time for time window selection: {time_tw.avg}")
                logging.info(f"time for similarity selection: {time_ss.avg}")
                logging.info(f"time for feature encoding: {time_ef.avg}")
                logging.info(f"time for forward process: {time_fw.avg}")
                break
        
        logging.info("Training has finished.")
        logging.info(f"Model of Epoch {best_epoch} checkpoint and metadata has been saved at {self.writer.log_dir}.")