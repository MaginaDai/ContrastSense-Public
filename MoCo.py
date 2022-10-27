# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from cProfile import label
from cmath import nan
from configparser import NoOptionError
from copy import deepcopy
from itertools import dropwhile
import numbers
import os
import logging
import pdb
from re import I
from tkinter import Y
from tkinter.messagebox import NO
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from DeepSense import DeepSense_encoder

from SupContrast import SupConLoss
from data_aug.preprocessing import UsersNum
from figure_plot.figure_plot import t_SNE_view
from judge import AverageMeter
from neg_select import PCA_torch

from simclr import MultiHeadedSelfAttention, LayerNorm, PositionWiseFeedForward
from utils import f1_cal, save_config_file, save_checkpoint, accuracy, CPC_evaluate, MoCo_evaluate
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, adjusted_rand_score
from sklearn.manifold import TSNE


class MoCo_model(nn.Module):
    def __init__(self, transfer=False, out_dim=512, classes=6, dims=32, classifier_dim=None, final_dim=8, momentum=0.9, drop=0.1, DAL=False, users_class=None):
        super(MoCo_model, self).__init__()
        self.DAL = DAL
        self.encoder = MoCo_encoder(dims=dims, momentum=momentum, drop=drop)
        if transfer:
            self.classifier = MoCo_classifier(classes=classes, dims=dims, classifier_dim=classifier_dim, final_dim=final_dim, drop=drop)
        else:
            self.projector = MoCo_projector(out_dim=out_dim)
        if self.DAL:
            if users_class is None:
                self.discriminator = MoCo_discriminator(out_dim=out_dim)
            else:
                self.discriminator = MoCo_discriminator(out_dim=users_class)
                # print(users_class)
        self.transfer = transfer
        

    def forward(self, x):
        h = self.encoder(x)
        if self.transfer:
            z = self.classifier(h)
        else:
            z = self.projector(h)

        if self.DAL:
            d = self.discriminator(h)
        else:
            d = None
        return z, d


class MoCo_classifier(nn.Module):
    def __init__(self, classes=6, dims=32, classifier_dim=None, final_dim=8, drop=0.1):
        super(MoCo_classifier, self).__init__()

        self.gru = torch.nn.GRU(dims, final_dim, num_layers=1, batch_first=True, bidirectional=True) #??? why decrease it....
        self.MLP = nn.Sequential(nn.Linear(in_features=3200, out_features=classifier_dim), # 1920 for 120
                                nn.ReLU(),
                                nn.Linear(in_features=classifier_dim, out_features=classes))
        self.dropout = torch.nn.Dropout(p=drop)
    
    def forward(self, h):
        self.gru.flatten_parameters()
        # h, _ = self.gru1(h)
        # h = self.dropout(h)
        h, _ = self.gru(h)
        h = self.dropout(h)
        h = h.reshape(h.shape[0], -1)
        h = self.MLP(h)
        return h

class MoCo_discriminator(nn.Module):
    def __init__(self, out_dim=32):
        super(MoCo_discriminator, self).__init__()
        self.discriminator = nn.Sequential(
                                GradientReversal(),
                                nn.Linear(6400, out_dim*4),
                                nn.ReLU(),
                                nn.Linear(out_dim*4, out_dim*2),
                                nn.ReLU(),
                                nn.Linear(out_dim*2, out_dim)
                            )
    
    def forward(self, h):
        h = h.reshape(h.shape[0], -1)
        h = self.discriminator(h)
        return h


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    


class MoCo_projector(nn.Module):
    def __init__(self, out_dim=512):
        super(MoCo_projector, self).__init__()
        self.linear1= torch.nn.Linear(in_features=6400, out_features=out_dim*4)  # 3840 for 120
        self.linear2 = torch.nn.Linear(in_features=out_dim*4, out_features=out_dim*2)
        self.linear3 = torch.nn.Linear(in_features=out_dim*2, out_features=out_dim)
        self.relu = torch.nn.ReLU()
    
    def forward(self, h):
        # keep the output layer constant with the SimCLR output
        h = h.reshape(h.shape[0], -1)
        h = self.relu(self.linear1(h))
        h = self.relu(self.linear2(h))  # add nonlinear / add linear
        z = self.linear3(h)
        return z


class MoCo_encoder(nn.Module):

    def __init__(self, dims=32, momentum=0.9, drop=0.1):
        super(MoCo_encoder, self).__init__()

        self.dropout = torch.nn.Dropout(p=drop)
        self.relu = torch.nn.ReLU()

        self.conv1_acc = torch.nn.Conv2d(1, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))
        self.conv1_gyro = torch.nn.Conv2d(1, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))
        # self.conv1_acc_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=momentum, affine=False)
        # self.conv1_gyro_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=momentum, affine=False)

        self.conv2_acc_1 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))
        self.conv2_acc_2 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))
        # self.conv2_acc_BN_1 =torch.nn.BatchNorm2d(num_features=dims, momentum=momentum, affine=False)
        # self.conv2_acc_BN_2 =torch.nn.BatchNorm2d(num_features=dims, momentum=momentum, affine=False)

        self.conv2_gyro_1 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))
        self.conv2_gyro_2 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))
        # self.conv2_gyro_BN_1 =torch.nn.BatchNorm2d(num_features=dims, momentum=momentum, affine=False)
        # self.conv2_gyro_BN_2 =torch.nn.BatchNorm2d(num_features=dims, momentum=momentum, affine=False)
        
        
        self.BN_acc = torch.nn.BatchNorm2d(num_features=dims, momentum=momentum, affine=False)
        self.BN_gyro = torch.nn.BatchNorm2d(num_features=dims, momentum=momentum, affine=False)

        self.conv3 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 3), stride=(1, 3), padding=(2, 0))
        # self.conv3_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=momentum, affine=False)

        self.conv4_1 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))
        self.conv4_2 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))
        # self.conv4_BN_1 =torch.nn.BatchNorm2d(num_features=dims, momentum=momentum, affine=False)
        # self.conv4_BN_2 =torch.nn.BatchNorm2d(num_features=dims, momentum=momentum, affine=False)

        self.conv5_1 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 2), stride=1, padding=(2, 0))
        # self.conv5_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=momentum, affine=False)

        self.attn = MultiHeadedSelfAttention(dims)
        self.proj = nn.Linear(dims, dims)
        self.norm1 = LayerNorm(dims)
        self.pwff = PositionWiseFeedForward(hidden_dim=dims, hidden_ff=dims*2)
        self.norm2 = LayerNorm(dims)


    def forward(self, x):

        # extract in-sensor information
        x_acc = self.conv1_acc(x[:, :, :, 0:3])
        # x_acc = self.conv1_acc_BN(x_acc)
        x_acc = self.dropout(self.relu(x_acc))
        # if torch.isnan(x_acc).any():
        #     pdb.set_trace()

        x_gyro = self.conv1_gyro(x[:, :, :, 3:])
        # x_gyro = self.conv1_gyro_BN(x_gyro)
        x_gyro = self.dropout(self.relu(x_gyro))
        # if torch.isnan(x_gyro).any():
        #     pdb.set_trace()

        # ResNet Arch for high-level information
        x1 = self.conv2_acc_1(x_acc)
        # x1 = self.conv2_acc_BN_1(x1)
        x1 = self.dropout(self.relu(x1))
        x1 = self.conv2_acc_2(x1)
        # x1 = self.conv2_acc_BN_2(x1)
        x_acc = self.dropout(self.relu(x_acc + x1))
        # if torch.isnan(x_acc).any():
        #     pdb.set_trace()

        x1 = self.conv2_gyro_1(x_gyro)
        # x1 = self.conv2_gyro_BN_1(x1)
        x1 = self.dropout(self.relu(x1))
        x1 = self.conv2_gyro_2(x1)
        # x1 = self.conv2_gyro_BN_2(x1)
        x_gyro = self.dropout(self.relu(x_gyro + x1))
        # if torch.isnan(x_gyro).any():
        #     pdb.set_trace()
        
        # # we need to normalize the data to make the features comparable
        x_acc = self.BN_acc(x_acc)
        x_gyro = self.BN_gyro(x_gyro)

        h = torch.cat((x_acc, x_gyro), dim=3)
        # extract intra-sensor information
        h = self.conv3(h)
        # h = self.conv3_BN(h)
        h = self.dropout(self.relu(h))
        # if torch.isnan(h).any():
        #     pdb.set_trace()

        # ResNet Arch for high-level information
        x1 = self.conv4_1(h)
        # x1 = self.conv4_BN_1(h)
        x1 = self.dropout(self.relu(x1))
        x1 = self.conv4_2(x1)
        # x1 = self.conv4_BN_2(h)
        h = self.dropout(self.relu(h + x1))
        # if torch.isnan(h).any():
        #     pdb.set_trace()

        h = self.conv5_1(h)
        # h = self.conv5_BN(h)
        h = self.dropout(self.relu(h))
        
        h = h.view(h.shape[0], h.shape[1], -1)
        h = h.permute(0, 2, 1)
        # if torch.isnan(h).any():
        #     pdb.set_trace()
        
        h = self.attn(h)
        h = self.norm1(h + self.proj(h))
        h = self.norm2(h + self.pwff(h))
        h = self.dropout(h)

        # if torch.isnan(h).any():
        #     pdb.set_trace()
        return h


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class MoCo_v1(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, device, transfer=False, out_dim=128, K=65536, m=0.999, T=0.07, T_labels=None, classes=6, dims=32,\
                 label_type=2, num_clusters=None, mol='MoCo', final_dim=32, momentum=0.9, drop=0.1, DAL=False, if_cross_entropy=False, users_class=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_v1, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.T_labels = T_labels
        self.label_type = label_type
        self.if_cross_entropy = if_cross_entropy
        self.DAL = DAL

        # create the encoders
        # num_classes is the output fc dimension
        if mol == 'MoCo':
            if self.if_cross_entropy:
                self.encoder_q = MoCo_model(transfer=transfer, out_dim=out_dim, classes=classes, dims=dims, 
                                            final_dim=final_dim, momentum=momentum, drop=drop, DAL=self.DAL, users_class=users_class)
                self.encoder_k = MoCo_model(transfer=transfer, out_dim=out_dim, classes=classes, dims=dims,
                                            final_dim=final_dim, momentum=momentum, drop=drop, DAL=self.DAL, users_class=users_class)
            else:
                self.encoder_q = MoCo_model(transfer=transfer, out_dim=out_dim, classes=classes, dims=dims, final_dim=final_dim, momentum=momentum, drop=drop, DAL=self.DAL)
                self.encoder_k = MoCo_model(transfer=transfer, out_dim=out_dim, classes=classes, dims=dims, final_dim=final_dim, momentum=momentum, drop=drop, DAL=self.DAL)
        elif mol == 'DeepSense':
            self.encoder_q = DeepSense_encoder(transfer=transfer, out_dim=out_dim, classes=classes, dims=dims)
            self.encoder_k = DeepSense_encoder(transfer=transfer, out_dim=out_dim, classes=classes, dims=dims)

        self.sup_loss = SupConLoss(device=device, if_cross_entropy=self.if_cross_entropy)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(out_dim, K))
        if users_class is not None:
            self.register_buffer("queue_dis", torch.randn(users_class, K))
        else:
            self.register_buffer("queue_dis", torch.randn(out_dim, K))
        self.register_buffer("queue_labels", torch.randint(0, 10, [K, label_type]))  # store label for SupCon
        self.register_buffer("queue_gt", torch.randint(0, 10, [K, 1])) # store label for visualization

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_dis_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_labels_ptr", torch.zeros(label_type, dtype=torch.long))
        self.register_buffer("queue_gt_ptr", torch.zeros(1, dtype=torch.long))

        self.queue = F.normalize(self.queue, dim=0)
        self.queue_dis = F.normalize(self.queue_dis, dim=0)
        if num_clusters is not None:
            self.cluster_centers = []

        self.tsne = TSNE(n_components=2, learning_rate='auto', init='random')

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels=None, gt=None, d=None):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        assert self.K % batch_size == 0  # for simplicity

        ptr = int(self.queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T  # modified
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

        if labels is not None:
            for i, l in enumerate(labels):
                ptr_label = int(self.queue_labels_ptr[i])
                self.queue_labels[ptr_label:ptr_label + batch_size, i] = l
                ptr_label = (ptr_label + batch_size) % self.K
                self.queue_labels_ptr[i] = ptr_label
        
        if gt is not None:
            ptr_gt = int(self.queue_gt_ptr)
            self.queue_gt[ptr_gt:ptr_gt + batch_size, 0] = gt
            ptr_gt = (ptr_gt + batch_size) % self.K
            self.queue_gt_ptr[0] = ptr_gt
        
        if d is not None:
            dis_ptr = int(self.queue_dis_ptr)
            # replace the keys at ptr (dequeue and enqueue)
            self.queue_dis[:, dis_ptr:dis_ptr + batch_size] = d.T  # modified
            dis_ptr = (dis_ptr + batch_size) % self.K  # move pointer
            self.queue_dis_ptr[0] = dis_ptr
            

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    def forward(self, sen_q, sen_k, labels, num_clusters, iter_tol, gt, if_plot=False, n_iter=None):
        """
        Input:
            sen_q: a batch of query sensors data
            sen_k: a batch of key sensors data
            labels: the label for nuisance suppression
            gt: the ground truth label for visualization
        Output:
            logits, targets
        """
        device = (torch.device('cuda')
            if sen_q.is_cuda
            else torch.device('cpu'))

        q, d_q = self.encoder_q(sen_q)  # queries: NxC
        q = F.normalize(q, dim=1)
        if d_q is not None:
            d_q = F.normalize(d_q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k, d_k = self.encoder_k(sen_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            if d_k is not None:
                d_k = F.normalize(d_k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        feature = torch.concat([q, self.queue.clone().detach().T], dim=0)  # the overall features rather than the dot product of features.  

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # targets: positive key indicators
        targets = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, labels, gt, d_k)
        
        if self.DAL: # we do / T when calculating the SupCon
            logits_labels = torch.einsum('nc,ck->nk', [d_q, self.queue_dis.clone().detach()])
        else:
            logits_labels = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        if num_clusters:
            h = torch.cat((q.detach(), self.queue.clone().T.detach()), 0).to(device)
            
            cluster_labels, self.cluster_centers, num_record, center_shift = kmeans(
                X=h, num_clusters=num_clusters, distance='cosine', 
                device=device, tol=iter_tol, tqdm_flag=False, cluster_centers=self.cluster_centers
            )
            # we set the q-k as the same class
            cluster_labels[q.shape[0]:2*q.shape[0]] = cluster_labels[:q.shape[0]]
            chs = calinski_harabasz_score(h.clone().cpu().detach().numpy(), cluster_labels.clone().cpu().detach().numpy())
            # drs = adjusted_rand_score(labels, cluster_labels)
            cluster_eval = chs
            cluster_loss = self.sup_loss(logits_labels/self.T, 
                                         labels=cluster_labels[:q.shape[0]],
                                         queue_labels=cluster_labels[q.shape[0]:])
        else:
            cluster_eval = None
            cluster_loss = None
            center_shift = None
        
        # if torch.isnan(q).any() or torch.isnan(k).any():
        #     pdb.set_trace()
        #     print("now")

        # if if_plot:
        #     with torch.no_grad():
        #         h = torch.cat((q.clone().cpu(), self.queue.clone().cpu().T), 0).numpy()
        #         gt_all = torch.cat((gt.clone().cpu(), self.queue_gt.clone().squeeze().cpu())).numpy()
        #         h_trans = self.tsne.fit_transform(h)
        #         t_SNE_view(h_trans, gt_all, n_iter)
                
        
        return logits, targets, logits_labels, cluster_eval, cluster_loss, center_shift, feature
    
    
    def supervised_CL(self, logits_labels=None, labels=None):
        if labels:
            loss = torch.zeros(self.label_type)
            for i in range(self.label_type):
                loss[i] = self.sup_loss(logits_labels/self.T_labels[i], labels=labels[i], queue_labels=self.queue_labels[:, i].view(-1, 1))
            return loss
        else:
            return None


class MoCo(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        if self.args.DAL and self.args.transfer:
            self.optimizer_DAL = kwargs['optimizer_DAL']
            self.scheduler_DAL = kwargs['scheduler_DAL']
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

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start MoCo training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")

        acc = 0
        best_epoch = 0
        best_acc = 0
        not_best_counter = 0
        best_loss = 1e6
        see_cluster_effect = False


        for epoch_counter in tqdm(range(self.args.epochs)):
            acc_batch = AverageMeter('acc_batch', ':6.2f')
            chs = AverageMeter('chs', ':6.2f')
            loss_batch = AverageMeter('loss_batch', ':6.5f')

            for sensor, labels in train_loader:
                if n_iter % 100 == 0 and n_iter != 0: # every 250 epoch produce one image.
                    see_cluster_effect = True
                else:
                    see_cluster_effect = False
                sensor = [t.to(self.args.device) for t in sensor]
                gt_label = labels[:, 0].to(self.args.device) # the first dim is motion labels
                sup_label = [labels[:, i + 1].to(self.args.device) for i in range(self.args.label_type)]  # the following dim are cheap labels
                with autocast(enabled=self.args.fp16_precision):
                    output, target, logits_labels, cluster_eval, cluster_loss, center_shift, feature = self.model(sensor[0], sensor[1], labels=sup_label, 
                                                                                                        num_clusters=self.args.num_clusters, 
                                                                                                        iter_tol=self.args.iter_tol,
                                                                                                        gt=gt_label, if_plot=see_cluster_effect,
                                                                                                        n_iter=n_iter)
                    if self.model.if_cross_entropy:
                        sup_loss = self.model.supervised_CL(logits_labels=feature, labels=sup_label)
                    else:
                        sup_loss = self.model.supervised_CL(logits_labels=logits_labels, labels=sup_label)
                    if cluster_loss is not None:
                        loss = cluster_loss
                    else:
                        loss = self.criterion(output, target)
                    ori_loss = loss.detach().clone()
                    if sup_loss is not None:
                        for i in range(len(sup_loss)):
                            if self.model.DAL:  # DAL use revearse layer so we just add them up.
                                loss += self.args.slr[i] * sup_loss[i]
                            else:
                                loss -= self.args.slr[i] * sup_loss[i]

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                acc = accuracy(output, target, topk=(1,))

                acc_batch.update(acc, sensor[0].size(0))
                loss_batch.update(loss, sensor[0].size(0))
                if cluster_eval is not None:
                    chs.update(cluster_eval, sensor[0].size(0))
                
                if n_iter % self.args.log_every_n_steps == 0 and n_iter != 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc', acc, global_step=n_iter)
                    self.writer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], global_step=n_iter)
                    if sup_loss is not None:
                        self.writer.add_scalar('ori_loss_{}'.format(i), ori_loss, global_step=n_iter)
                        for i in range(len(sup_loss)):
                            self.writer.add_scalar('sup_loss_{}'.format(i), sup_loss[i], global_step=n_iter)
                    
                    if cluster_eval is not None:
                        self.writer.add_scalar('chs', cluster_eval, global_step=n_iter)
                        self.writer.add_scalar('center_shift', center_shift, global_step=n_iter)

                n_iter += 1
            
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
            # in current setting, it is not meaningful
            if epoch_counter >= 10:
                self.scheduler.step()
            
            if epoch_counter == 0:
                continue  # the first epoch would not have record.
            log_str = f"Epoch: {epoch_counter} Loss: {loss_batch.avg} accuracy: {acc_batch.avg} "
            if sup_loss is not None:
                log_str += f"ori_Loss :{ori_loss} "
                for i in range(len(sup_loss)):
                    log_str += f"sup_loss_{i}: {sup_loss[i]} "
            if cluster_eval is not None:
                log_str += f"chs: {chs.avg} center_shift: {center_shift}"
            logging.debug(log_str)

            if best_acc > 99 and epoch_counter >= 50:
                break  # early stop

            if not_best_counter >= 200 or best_acc > 99:
                print(f"early stop at {epoch_counter}")
                break
            
        
        logging.info("Training has finished.")
        logging.info(f"Model of Epoch {best_epoch} checkpoint and metadata has been saved at {self.writer.log_dir}.")

    def transfer_train(self, tune_loader, val_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter_train = 0
        logging.info(f"Start MoCo fine-tuning head for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")

        acc = 0
        f1 = 0
        best_epoch = 0

        if self.args.resume:
            best_f1 = self.args.best_f1
            best_acc = self.args.best_acc
        else:
            best_f1 = 0
            best_acc = 0

        for epoch_counter in tqdm(range(self.args.epochs)):
            acc_batch = AverageMeter('acc_batch', ':6.2f')
            f1_batch = AverageMeter('f1_batch', ':6.2f')
            if self.args.if_fine_tune:
                self.model.train()
            else:  
                """
                Switch to eval mode:
                Under the protocol of linear classification on frozen features/models,
                it is not legitimate to change any part of the pre-trained model.
                BatchNorm in train mode may revise running mean/std (even if it receives
                no gradient), which are part of the model parameters too.
                """
                self.model.eval()
                self.model.classifier.train()
                
            for sensor, target in tune_loader:

                sensor = sensor.to(self.args.device)
                target = target[:, 0].to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    logits, _ = self.model(sensor)
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

            if self.args.if_val:
                val_acc, val_f1 = MoCo_evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=val_loader)
            else:
                val_acc = 0

            is_best = val_f1 > best_f1

            if epoch_counter >= 10:  # only after the first 10 epochs, the best_f1/acc is updated.
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
            self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter} Loss: {loss} acc: {acc_batch.avg: .3f}/{val_acc: .3f} f1: {f1_batch.avg: .3f}/{val_f1: .3f}")

        logging.info("Fine-tuning has finished.")
        logging.info(f"best eval f1 is {best_f1} at {best_epoch}.")

        print('best eval f1 is {} for {}'.format(best_f1, self.args.name))

    def test_performance(self, best_model_dir, test_loader):
        checkpoint = torch.load(best_model_dir, map_location="cpu")
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        test_acc, test_f1 = MoCo_evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=test_loader)
        logging.info(f"test f1 is {test_f1}.")
        logging.info(f"test acc is {test_acc}.")

        print('test f1 is {} for {}'.format(test_f1, self.args.name))
        print('test acc is {} for {}'.format(test_acc, self.args.name))
    
    def transfer_train_DAL(self, tune_loader, val_loader, train_loader):
        """
        train loader is for domain adversial learning
        we only consider fine-tune + DAL. Linear Evaluation is not considered for this type of learning. 
        """
        assert self.args.if_fine_tune is True
        
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter_train = 0
        n_iter_train_DAL = 0
        logging.info(f"Start MoCo fine-tuning head for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")

        acc = 0
        f1 = 0
        best_epoch = 0

        if self.args.resume:
            best_f1 = self.args.best_f1
            best_acc = self.args.best_acc
        else:
            best_f1 = 0
            best_acc = 0

        for epoch_counter in tqdm(range(self.args.epochs)):
            acc_batch = AverageMeter('acc_batch', ':6.2f')
            f1_batch = AverageMeter('f1_batch', ':6.2f')
            self.model.train()

            for sensor, target in tune_loader:

                sensor = sensor.to(self.args.device)
                target = target[:, 0].to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    h = self.model.encoder(sensor)
                    logits = self.model.classifier(h)
                    loss = self.criterion(logits, target)  # fine-tune for HAR

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
            
            for sensor, target in train_loader:

                sensor = sensor.to(self.args.device)
                target = target[:, 1].to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    h = self.model.encoder(sensor)
                    logits = self.model.discriminator(h)
                    loss = self.criterion(logits, target)  # train for users discrimination
                
                if n_iter_train % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss_DAL', loss, global_step=n_iter_train_DAL)
                    self.writer.add_scalar('lr_DAL', self.scheduler_DAL.get_last_lr()[0], global_step=n_iter_train_DAL)
                n_iter_train_DAL += 1
                self.optimizer_DAL.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer_DAL)
                scaler.update()

            if self.args.if_val:
                val_acc, val_f1 = MoCo_evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=val_loader)
            else:
                val_acc = 0

            is_best = val_f1 > best_f1

            if epoch_counter >= 10:  # only after the first 10 epochs, the best_f1/acc is updated.
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
            self.scheduler.step()
            self.scheduler_DAL.step()
            logging.debug(f"Epoch: {epoch_counter} Loss: {loss} acc: {acc_batch.avg: .3f}/{val_acc: .3f} f1: {f1_batch.avg: .3f}/{val_f1: .3f}")

        logging.info("Fine-tuning has finished.")
        logging.info(f"best eval f1 is {best_f1} at {best_epoch}.")

        print('best eval f1 is {} for {}'.format(best_f1, self.args.name))

