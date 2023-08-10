# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from cProfile import label
from cmath import nan
from configparser import NoOptionError
from copy import deepcopy
from itertools import dropwhile
import numbers
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

mol='MoCo'

class MoCo_model(nn.Module):
    def __init__(self, transfer=False, out_dim=256, classes=6, dims=32, classifier_dim=1024, final_dim=8, momentum=0.9, drop=0.1, DAL=False, users_class=None, SCL=False, modal='imu'):
        super(MoCo_model, self).__init__()
        self.DAL = DAL
        self.modal = modal
        self.transfer = transfer

        if self.modal == 'imu':
            self.encoder = MoCo_encoder(dims=dims, momentum=momentum, drop=drop)
        elif self.modal == 'emg':
            self.encoder = MoCo_encoder_for_emg_v2(dims=dims, momentum=momentum, drop=drop)
        elif self.modal == 'eeg':
            self.encoder = SACLEncoder_CNN_block(num_channels=1, temporal_len=3000)
        else:
            NotADirectoryError
        
        if transfer:
            self.classifier = MoCo_classifier(classes=classes, dims=dims, classifier_dim=classifier_dim, final_dim=final_dim, drop=drop, modal=self.modal)
            if SCL:
                self.projector = MoCo_projector(out_dim=out_dim, modal=self.modal)
        else:
            self.projector = MoCo_projector(out_dim=out_dim, modal=self.modal)
        
        if self.DAL:
            if users_class is None:
                self.discriminator = MoCo_discriminator(out_dim=out_dim, modal=self.modal)
            else:
                self.discriminator = MoCo_discriminator(out_dim=users_class, modal=self.modal)
                # print(users_class)
        
        

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
    def __init__(self, classes=6, dims=32, classifier_dim=1024, final_dim=8, drop=0.1, modal='imu'):
        super(MoCo_classifier, self).__init__()

        if modal == 'imu':
            feature_num = 3200
            self.gru = torch.nn.GRU(dims, final_dim, num_layers=1, batch_first=True, bidirectional=True)
        elif modal == 'emg':
            feature_num = 1024
            # feature_num = 832
            self.gru = torch.nn.GRU(16, final_dim, num_layers=1, batch_first=True, bidirectional=True)
        elif modal == 'eeg':
            feature_num = 2976
            classifier_dim = 256
            self.gru = torch.nn.GRU(4, final_dim, num_layers=1, batch_first=True, bidirectional=True)
        else:
            NotADirectoryError
    
        
        self.MLP = nn.Sequential(nn.Linear(in_features=feature_num, out_features=classifier_dim), # 1920 for 120
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
    def __init__(self, out_dim=32, modal='imu'):
        super(MoCo_discriminator, self).__init__()
        if modal == 'imu':
            feature_num = 6400
        elif modal == 'emg':
            feature_num = 1024
            # feature_num = 2912
        else:
            NotADirectoryError

        self.discriminator = nn.Sequential(
                                GradientReversal(),
                                nn.Linear(feature_num, out_dim*4),
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
    def __init__(self, out_dim=512, modal='imu'):
        super(MoCo_projector, self).__init__()
        if modal == 'imu':
            feature_num = 6400
        elif modal == 'emg':
            feature_num = 1024
            # feature_num = 2912
        elif modal == 'eeg':
            feature_num = 744
        else:
            NotADirectoryError

        self.linear1= torch.nn.Linear(in_features=feature_num, out_features=out_dim*4)  # 3840 for 120
        self.linear2 = torch.nn.Linear(in_features=out_dim*4, out_features=out_dim*2)
        self.linear3 = torch.nn.Linear(in_features=out_dim*2, out_features=out_dim)
        self.relu = torch.nn.ReLU()
    
    def forward(self, h):
        # keep the output layer constant with the SimCLR output
        h = h.reshape(h.shape[0], -1)
        h = self.relu(self.linear1(h))
        h = self.relu(self.linear2(h))
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
    

class MoCo_encoder_for_emg(nn.Module):
    def __init__(self, dims=32, momentum=0.9, drop=0.1):
        super(MoCo_encoder_for_emg, self).__init__()

        self.dropout = torch.nn.Dropout(p=drop)
        self.relu = torch.nn.ReLU()

        self.per_channel_conv = nn.Sequential(
            torch.nn.Conv2d(1, dims, kernel_size=(5, 1), stride=1, padding=(2, 0)),
            self.relu,
            self.dropout,
            torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0)),
            self.relu,
            self.dropout,
            torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0)),
        )
        self.BN_1 = torch.nn.BatchNorm2d(num_features=dims, momentum=momentum, affine=False)

        self.cross_channel_conv = nn.Sequential(
            torch.nn.Conv2d(dims, dims, kernel_size=(5, 4), stride=(1, 4), padding=(2, 0)),  # We follow the design for IMU, the former 4 channels and the later 4 channels are fused seperately.
            self.relu,
            self.dropout,
        )

        self.per_channel_conv_2 = nn.Sequential(
            torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0)),
            self.relu,
            self.dropout,
            torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0)),
        )

        self.cross_channel_conv_2 = nn.Sequential(
            torch.nn.Conv2d(dims, dims, kernel_size=(5, 2), stride=1, padding=(2, 0)),  # We follow the design for IMU, the former 4 channels and the later 4 channels are fused seperately.
            self.relu,
            self.dropout,
        )

        self.attn = MultiHeadedSelfAttention(dims)
        self.proj = nn.Linear(dims, dims)
        self.norm1 = LayerNorm(dims)
        self.pwff = PositionWiseFeedForward(hidden_dim=dims, hidden_ff=dims*2)
        self.norm2 = LayerNorm(dims)



    def forward(self, x):
        h = self.per_channel_conv(x)
        h = self.dropout(self.relu(h + x))
        h = self.BN_1(h)

        h = self.cross_channel_conv(h)

        h1 = self.per_channel_conv_2(h)
        h = self.dropout(self.relu(h1 + h))

        h = self.cross_channel_conv_2(h)

        h = h.view(h.shape[0], h.shape[1], -1)
        h = h.permute(0, 2, 1)
        # if torch.isnan(h).any():
        #     pdb.set_trace()
        
        h = self.attn(h)
        h = self.norm1(h + self.proj(h))
        h = self.norm2(h + self.pwff(h))
        h = self.dropout(h)
        return h
    

class MoCo_encoder_for_emg_v2(nn.Module):  
    ## keep align with lysseCoteAllard/MyoArmbandDataset
    ## at 62b886fc7014aeb81af65d77affedadf40de684c (github.com)
    def __init__(self, dims=32, momentum=0.9, drop=0.1):
        super(MoCo_encoder_for_emg_v2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, dims, kernel_size=(5, 3)),
            nn.BatchNorm2d(dims),
            nn.PReLU(dims),
            nn.Dropout2d(.5),
            nn.MaxPool2d(kernel_size=(3, 1)),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(dims, dims*2, kernel_size=(5, 3)),
            nn.BatchNorm2d(dims*2),
            nn.PReLU(dims*2),
            nn.Dropout2d(.5),
            nn.MaxPool2d(kernel_size=(3, 1)),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x.view(x.shape[0], x.shape[1], -1)  # to keep align with IMU encoder


class MoCo_encoder_for_emg_v3(nn.Module):
    ## keep align with ConSSL
    def __init__(self, dims=32, momentum=0.9, drop=0.1):
        super(MoCo_encoder_for_emg_v3, self).__init__()

        A = np.ones([8, 8])

        self.l1 = TCN_GCN_unit(1, 4, A)
        self.l2 = TCN_GCN_unit(4, 8, A)
        self.l3 = TCN_GCN_unit(8, 8, A)
        self.l4 = TCN_GCN_unit(8, 7, A)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = torch.permute(x, (0, 2, 1, 3))
        return x.reshape(x.shape[0], x.shape[1], -1)  # to keep align with IMU encoder
    
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
    def __init__(self, args, transfer=False, classes=6, out_dim=256, users_class=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_v1, self).__init__()

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

        self.hard_sample = args.hard
        self.sample_ratio = args.sample_ratio
        self.last_ratio = args.last_ratio
        self.hard_record = args.hard_record
        self.time_window = args.time_window
        self.scale_ratio = args.scale_ratio 
        

        # create the encoders
        # num_classes is the output fc dimension
        if mol == 'MoCo':
            if self.if_cross_entropy:
                self.encoder_q = MoCo_model(transfer=transfer, classes=classes, DAL=self.DAL, users_class=users_class, modal=modal)
                self.encoder_k = MoCo_model(transfer=transfer, DAL=self.DAL, users_class=users_class, modal=modal)
            else:
                self.encoder_q = MoCo_model(transfer=transfer, DAL=self.DAL, modal=self.modal)
                self.encoder_k = MoCo_model(transfer=transfer, DAL=self.DAL, modal=self.modal)
        
        elif mol == 'DeepSense':
            self.encoder_q = DeepSense_encoder()
            self.encoder_k = DeepSense_encoder()

        self.sup_loss = SupConLoss(device=device, if_cross_entropy=self.if_cross_entropy)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(out_dim, self.K))
        if users_class is not None:
            self.register_buffer("queue_dis", torch.randn(users_class, self.K))
        else:
            self.register_buffer("queue_dis", torch.randn(out_dim, self.K))
        
        self.register_buffer("queue_labels", torch.randint(0, 10, [self.K, 1]))  # store label for SupCon
        self.register_buffer("queue_gt", torch.randint(0, 10, [self.K, 1])) # store label for visualization
        self.register_buffer("queue_time_labels", torch.randint(0, 10, [self.K, 1]))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_dis_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_labels_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_gt_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_time_ptr", torch.zeros(1, dtype=torch.long))

        self.queue = F.normalize(self.queue, dim=0)
        self.queue_dis = F.normalize(self.queue_dis, dim=0)

        self.tsne = TSNE(n_components=2, learning_rate='auto', init='random')

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels=None, gt=None, d=None, time_label=None):
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
        
        if time_label is not None:
            ptr_time = int(self.queue_time_ptr)
            self.queue_time_labels[ptr_time:ptr_time + batch_size, 0] = time_label[0]
            ptr_time = (ptr_time + batch_size) % self.K
            self.queue_time_ptr[0] = ptr_time

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
    

    def forward(self, sen_q, sen_k, domain_label, gt, time_label):
        """
        Input:
            sen_q: a batch of query sensors data
            sen_k: a batch of key sensors data
            domain_label: the label for nuisance suppression
            gt: the ground truth label for visualization
        Output:
            logits, targets
        """
        device = (torch.device('cuda')
            if sen_q.is_cuda
            else torch.device('cpu'))
        similarity_across_domains = 0
        
        hardest_related_info = [0, 0, 0, 0, 0]
        mean_same_class_ratio = 0
        mean_same_domain_ratio = 0
        mean_same_class_same_domain = 0

        start_time = time.time()
        q, d_q = self.encoder_q(sen_q)  # queries: NxC
        q = F.normalize(q, dim=1)
        end_time = time.time()
        hardest_related_info[4] = end_time - start_time
        
        # if d_q is not None:
        #     d_q = F.normalize(d_q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k, d_k = self.encoder_k(sen_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            # if d_k is not None:
            #     d_k = F.normalize(d_k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  ## we further improve this step

        if self.hard_record:
            sim_wih_other_domain = l_neg.clone().detach()
            _, indices = torch.sort(sim_wih_other_domain, dim=1, descending=True)
            num_eliminate = int(l_neg.shape[1] * self.sample_ratio)

            rows = torch.arange(l_neg.shape[0]).unsqueeze(-1)
            que_labels = self.queue_gt.T.expand(l_neg.shape[0], l_neg.shape[1])
            labels_of_eliminated = que_labels[rows, indices[:, :num_eliminate]]
            mask_same_class = torch.eq(gt[0].contiguous().view(-1, 1), labels_of_eliminated)
            same_class_ratio = mask_same_class.sum(dim=1)/num_eliminate * 100

            que_domain_labels = self.queue_labels.T.expand(l_neg.shape[0], l_neg.shape[1])
            domain_labels_of_eliminated = que_domain_labels[rows, indices[:, :num_eliminate]]
            mask_same_domain = torch.eq(domain_label[0].contiguous().view(-1, 1), domain_labels_of_eliminated)
            same_domain_ratio = mask_same_domain.sum(dim=1)/num_eliminate * 100
            same_class_same_domain = torch.logical_and(mask_same_class, mask_same_domain).sum(dim=1) / num_eliminate * 100
            
            mean_same_class_ratio = same_class_ratio.mean()
            mean_same_domain_ratio = same_domain_ratio.mean()
            mean_same_class_same_domain = same_class_same_domain.mean()
            hardest_related_info = [mean_same_class_ratio, mean_same_domain_ratio, mean_same_class_same_domain]
        
        # negative logits: NxK
        if self.hard_sample:
            #### v10 domain-wise sorting + time window
            if self.last_ratio < 1.0:  ## if last_ratio >= 1, then we dont apple this simplist elimination. 
                start_time = time.time()
                domains_in_queues = torch.unique(self.queue_labels.clone().detach()).contiguous().view(-1, 1)
                domain_queues_mask = torch.eq(domains_in_queues, self.queue_labels.T).bool().to(device)
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
                queue_time = self.queue_time_labels.T.expand(l_neg.shape[0], l_neg.shape[1])
                mask_low = low_boundary < queue_time # (NxQ) = label of sen_q (Nx1) x labels of queue (Qx1).T
                mask_high = queue_time < high_boundary
                mask = torch.logical_and(mask_low, mask_high)
                l_neg[mask] = -torch.inf    
                hardest_related_info[0] = mask.sum(1).float().mean()
                
                end_time = time.time()
                hardest_related_info[3] = end_time - start_time

            

            #### v9 domain-wise threshold + time window
            # domains_in_queues = torch.unique(self.queue_labels.clone().detach()).contiguous().view(-1, 1)
            # domain_of_quarys = torch.unique(domain_label[0]).contiguous().view(-1, 1)

            # num_domains = len(domains_in_queues)
            # num_domain_of_quarys = len(domain_of_quarys)

            # domain_queues_mask = torch.eq(domains_in_queues, self.queue_labels.T).bool().to(device)
            # domain_representations = torch.vstack([torch.sum(self.queue[:, domain_queues_mask[i]], dim=1)/torch.sum(domain_queues_mask[i]) for i in range(num_domains)]).to(device)
            # domain_representations /= torch.norm(domain_representations, dim=1).view(-1, 1)
            # similarity_across_domains = torch.matmul(domain_representations, domain_representations.T) / self.scale_ratio
            # mask = torch.diag(torch.ones(num_domains)).bool()
            # similarity_across_domains[mask] = -torch.inf
            # similarity_across_domains = torch.exp(similarity_across_domains)/torch.exp(similarity_across_domains).sum(dim=1)
            # similarity_across_domains[mask] = torch.inf # don't eliminate samples in the own domains. 
            
            # domain_not_included_in_queues = [i for i in domain_of_quarys if i not in domains_in_queues]  # find the domains not included in the domain queues
            # if len(domain_not_included_in_queues) != 0:
            #     domain_not_included_in_queues = torch.vstack(domain_not_included_in_queues)
            #     append_similarity = torch.ones(len(domain_not_included_in_queues), num_domains).to(device) * 1 / num_domains  # the new domains share the same similarity across all domains
            #     similarity_across_domains = torch.vstack([similarity_across_domains, append_similarity])  ## combine both matrix  
            #     all_domains_considered = torch.vstack([domains_in_queues, domain_not_included_in_queues])
            # else:
            #     all_domains_considered = domains_in_queues

            # scaling_for_each_domain = 1./(1 + similarity_across_domains)  # calculate the scaling factor
            # # scaling_for_each_domain[torch.ones(num_domains).long(), torch.ones(num_domains).long()] = 0  
            # neg_for_sampling = l_neg.clone().detach()

            # for j, domain_for_compare in enumerate(domains_in_queues):
            #     key_in_domain_j = domain_queues_mask[j].repeat(neg_for_sampling.shape[0], 1)
            #     domain_queue_j = neg_for_sampling[key_in_domain_j].view(neg_for_sampling.shape[0], -1)
            #     avg_similarity = torch.mean(domain_queue_j, dim=1)

            #     idx_for_domains = torch.nonzero(torch.eq(all_domains_considered.view(1, -1), domain_label[0].unsqueeze(1)))
            #     threshold_for_domain_j = avg_similarity * scaling_for_each_domain[idx_for_domains[:, 1].long(), j * torch.ones(len(idx_for_domains)).long()]
                
            #     idx_to_eliminate = domain_queue_j < threshold_for_domain_j.unsqueeze(-1)
            #     position = torch.where(domain_queues_mask[j] == True)[0].repeat(neg_for_sampling.shape[0], 1)

            #     rows = torch.arange(neg_for_sampling.shape[0]).unsqueeze(-1)
            #     masks_for_domain_j = torch.zeros(neg_for_sampling.shape).bool().to(device)
            #     masks_for_domain_j[rows, position] = idx_to_eliminate
            #     l_neg[masks_for_domain_j] = -torch.inf
            
            # low_boundary = time_label[0].contiguous().view(-1, 1) - self.time_window
            # high_boundary = time_label[0].contiguous().view(-1, 1) + self.time_window
            # queue_time = self.queue_time_labels.T.expand(l_neg.shape[0], l_neg.shape[1])
            # mask_low = low_boundary < queue_time # (NxQ) = label of sen_q (Nx1) x labels of queue (Qx1).T
            # mask_high = queue_time < high_boundary
            # mask = torch.logical_and(mask_low, mask_high)
            # l_neg[mask] = -torch.inf

            # hardest_related_info[0] = mask.sum(1).float().mean()


            #### v8 domain-wise threshold
            # measure similarity across domains
            # domains_in_queues = torch.unique(self.queue_labels.clone().detach()).contiguous().view(-1, 1)
            # domain_of_quarys = torch.unique(domain_label[0]).contiguous().view(-1, 1)

            # num_domains = len(domains_in_queues)
            # num_domain_of_quarys = len(domain_of_quarys)

            # domain_queues_mask = torch.eq(domains_in_queues, self.queue_labels.T).bool().to(device)
            # domain_representations = torch.vstack([torch.sum(self.queue[:, domain_queues_mask[i]], dim=1)/torch.sum(domain_queues_mask[i]) for i in range(num_domains)]).to(device)
            # domain_representations /= torch.norm(domain_representations, dim=1).view(-1, 1)
            # similarity_across_domains = torch.matmul(domain_representations, domain_representations.T)
            # mask = torch.diag(torch.ones(num_domains)).bool()
            # similarity_across_domains[mask] = -torch.inf
            # similarity_across_domains = torch.exp(similarity_across_domains)/torch.exp(similarity_across_domains).sum(dim=1)
            # similarity_across_domains[mask] = torch.inf # don't eliminate samples in the own domains. 
            
            # domain_not_included_in_queues = [i for i in domain_of_quarys if i not in domains_in_queues]  # find the domains not included in the domain queues
            # if len(domain_not_included_in_queues) != 0:
            #     domain_not_included_in_queues = torch.vstack(domain_not_included_in_queues)
            #     append_similarity = torch.ones(len(domain_not_included_in_queues), num_domains).to(device) * 1 / num_domains  # the new domains share the same similarity across all domains
            #     similarity_across_domains = torch.vstack([similarity_across_domains, append_similarity])  ## combine both matrix  
            #     all_domains_considered = torch.vstack([domains_in_queues, domain_not_included_in_queues])
            # else:
            #     all_domains_considered = domains_in_queues

            # scaling_for_each_domain = 1./(1 + self.scale_ratio * similarity_across_domains)  # calculate the scaling factor
            # # scaling_for_each_domain[torch.ones(num_domains).long(), torch.ones(num_domains).long()] = 0  
            # neg_for_sampling = l_neg.clone().detach()

            # for j, domain_for_compare in enumerate(domains_in_queues):
            #     key_in_domain_j = domain_queues_mask[j].repeat(neg_for_sampling.shape[0], 1)
            #     domain_queue_j = neg_for_sampling[key_in_domain_j].view(neg_for_sampling.shape[0], -1)
            #     avg_similarity = torch.mean(domain_queue_j, dim=1)

            #     idx_for_domains = torch.nonzero(torch.eq(all_domains_considered.view(1, -1), domain_label[0].unsqueeze(1)))
            #     threshold_for_domain_j = avg_similarity * scaling_for_each_domain[idx_for_domains[:, 1].long(), j * torch.ones(len(idx_for_domains)).long()]
                
            #     idx_to_eliminate = domain_queue_j < threshold_for_domain_j.unsqueeze(-1)
            #     position = torch.where(domain_queues_mask[j] == True)[0].repeat(neg_for_sampling.shape[0], 1)

            #     rows = torch.arange(neg_for_sampling.shape[0]).unsqueeze(-1)
            #     masks_for_domain_j = torch.zeros(neg_for_sampling.shape).bool().to(device)
            #     masks_for_domain_j[rows, position] = idx_to_eliminate
            #     l_neg[masks_for_domain_j] = -torch.inf

            #### v7 time idx information + domain-wise easiest elimination
            # num_of_sampled = int(self.queue.shape[1] * self.last_ratio)
            # # measure similarity across domains
            # domains = torch.unique(self.queue_labels.clone().detach()).contiguous().view(-1, 1)
            # num_domains = len(domains)

            # domain_queues_mask = torch.eq(domains, self.queue_labels.T).bool().to(device)
            # domain_representations = torch.vstack([torch.sum(self.queue[:, domain_queues_mask[i]], dim=1)/torch.sum(domain_queues_mask[i]) for i in range(num_domains)]).to(device)
            # domain_representations /= torch.norm(domain_representations, dim=1).view(-1, 1)
            # similarity_across_domains = torch.matmul(domain_representations, domain_representations.T)
            # mask = torch.diag(torch.ones(num_domains)).bool()
            # similarity_across_domains[mask] = -torch.inf
            # similarity_across_domains = torch.exp(similarity_across_domains)/torch.exp(similarity_across_domains).sum(dim=1)
            # preserved_samples_per_domain = similarity_across_domains.mul(num_of_sampled).int()
            
            # sampled based on domains
            # for i, domain in enumerate(domains):
            #     idx_query_in_domain_i = torch.eq(domain_label[0].contiguous(), domain)
            #     query_in_domain_i = l_neg[idx_query_in_domain_i]
            #     for j, domain_for_compare in enumerate(domains):
            #         if i == j or 0 in query_in_domain_i.size():
            #             continue
            #         key_in_domain_j = domain_queues_mask[j].view(1, -1).repeat(query_in_domain_i.shape[0], 1)
            #         domain_queue_j = query_in_domain_i[key_in_domain_j].view(sum(idx_query_in_domain_i), -1)
            #         _, indices = torch.sort(domain_queue_j, dim=1, descending=True)
            #         idx_to_eliminate = indices[:, preserved_samples_per_domain[i, j]:]

            #         position = torch.where(domain_queues_mask[j] == True)[0].repeat(indices.shape[0], 1)
            #         idx = torch.where(idx_query_in_domain_i)[0]
                    
            #         rows = torch.arange(position.shape[0]).unsqueeze(-1)
            #         l_neg[idx[rows], position[rows, idx_to_eliminate]] = -torch.inf


            # low_boundary = time_label[0].contiguous().view(-1, 1) - self.time_window
            # high_boundary = time_label[0].contiguous().view(-1, 1) + self.time_window
            # queue_time = self.queue_time_labels.T.expand(l_neg.shape[0], l_neg.shape[1])
            # mask_low = low_boundary < queue_time # (NxQ) = label of sen_q (Nx1) x labels of queue (Qx1).T
            # mask_high = queue_time < high_boundary
            # mask_both = torch.logical_and(mask_low, mask_high)
            # l_neg[mask_both] = -torch.inf

            # hardest_related_info[0] = mask_both.sum(1).float().mean()


            #### v6 eliminate using time idx information
            # low_boundary = time_label[0].contiguous().view(-1, 1) - self.time_window
            # high_boundary = time_label[0].contiguous().view(-1, 1) + self.time_window
            # queue_time = self.queue_time_labels.T.expand(l_neg.shape[0], l_neg.shape[1])
            # mask_low = low_boundary < queue_time # (NxQ) = label of sen_q (Nx1) x labels of queue (Qx1).T
            # mask_high = queue_time < high_boundary
            # mask = torch.logical_and(mask_low, mask_high)
            # l_neg[mask] = -torch.inf

            # hardest_related_info[0] = mask.sum(1).float().mean()

            # #### v5 no domain-wise but semi-hard
            # mask = torch.eq(domain_label[0].contiguous().view(-1, 1), self.queue_labels.T).bool().to(device)  # (NxQ) = label of sen_q (Nx1) x labels of queue (Qx1).T
            # sample_mask = torch.zeros(l_neg.shape).bool().to(device)
            # sample_mask[mask] = True

            # num_eliminate = int(l_neg.shape[1] * self.sample_ratio)

            # sim_wih_other_domain = l_neg.clone().detach()
            # _, indices = torch.sort(sim_wih_other_domain, dim=1, descending=True)

            # rows = torch.arange(l_neg.shape[0]).unsqueeze(-1)
            # sample_mask[rows, indices[:, :num_eliminate]] = False  ## eliminate the hardest
            # mask[rows, indices[:, :num_eliminate]] = True ## when eliminate simple in the other domains, those should not be considered.

            # num_mask = torch.sum(~mask, dim=1)
            # domain_wise_len = (num_mask * self.last_ratio).int()

            # sim_wih_other_domain[mask] = -torch.inf  # then sample within the same domain would be -inf

            # _, indices = torch.sort(sim_wih_other_domain, dim=1, descending=True)
    
            # for i in range(indices.shape[0]):
            #     sample_mask[i, indices[i, :domain_wise_len[i]]] = True  ## in this case, the hardest sample from the self domain will not be eliminated!!!
            
            # l_neg[~sample_mask] = -torch.inf

            

            ## v4 lets further add the hard sampling     

            # ## eliminate simplest by domains
            # num_of_sampled = int(self.queue.shape[1] * self.last_ratio)
            # # measure similarity across domains
            # domains = torch.unique(self.queue_labels.clone().detach()).contiguous().view(-1, 1)
            # num_domains = len(domains)

            # domain_queues_mask = torch.eq(domains, self.queue_labels.T).bool().to(device)
            # domain_representations = torch.vstack([torch.sum(self.queue[:, domain_queues_mask[i]], dim=1)/torch.sum(domain_queues_mask[i]) for i in range(num_domains)]).to(device)
            # domain_representations /= torch.norm(domain_representations, dim=1).view(-1, 1)
            # similarity_across_domains = torch.matmul(domain_representations, domain_representations.T)
            # mask = torch.diag(torch.ones(num_domains)).bool()
            # similarity_across_domains[mask] = -torch.inf
            # similarity_across_domains = torch.exp(similarity_across_domains)/torch.exp(similarity_across_domains).sum(dim=1)
            # preserved_samples_per_domain = similarity_across_domains.mul(num_of_sampled).int()
            
            # # sampled based on domains
            # for i, domain in enumerate(domains):
            #     idx_query_in_domain_i = torch.eq(domain_label[0].contiguous(), domain)
            #     query_in_domain_i = l_neg[idx_query_in_domain_i]
            #     for j, domain_for_compare in enumerate(domains):
            #         if i == j or 0 in query_in_domain_i.size():
            #             continue
            #         key_in_domain_j = domain_queues_mask[j].view(1, -1).repeat(query_in_domain_i.shape[0], 1)
            #         domain_queue_j = query_in_domain_i[key_in_domain_j].view(sum(idx_query_in_domain_i), -1)
            #         _, indices = torch.sort(domain_queue_j, dim=1, descending=True)
            #         idx_to_eliminate = indices[:, preserved_samples_per_domain[i, j]:]

            #         position = torch.where(domain_queues_mask[j] == True)[0].repeat(indices.shape[0], 1)
            #         idx = torch.where(idx_query_in_domain_i)[0]
                    
            #         rows = torch.arange(position.shape[0]).unsqueeze(-1)
            #         l_neg[idx[rows], position[rows, idx_to_eliminate]] = -torch.inf

            # ### eliminate hardest
            # sim_wih_other_domain = l_neg.clone().detach().to(device)
            # _, indices = torch.sort(sim_wih_other_domain, dim=1, descending=True)
            # mask = torch.zeros(l_neg.shape).bool().to(device)
            # num_eliminate = int(l_neg.shape[1] * self.sample_ratio)

            # rows = torch.arange(indices.size(0)).unsqueeze(-1)
            # mask[rows, indices[:, :num_eliminate]] = True ## Top r% are eliminated. They are from the same class.
            # l_neg[mask] = -torch.inf


            #### v3 is proved to be effective + 2% F1 on average
            # sim_wih_other_domain = l_neg.clone().detach()
            # _, indices = torch.sort(sim_wih_other_domain, dim=1, descending=True)
            # mask = torch.ones(l_neg.shape).bool().to(device)
            # num_eliminate = int(l_neg.shape[1] * self.sample_ratio)
            # rows = torch.arange(l_neg.shape[0]).unsqueeze(-1)
            # mask[rows, indices[:, :num_eliminate]] = False ## Top r% are eliminated.
            # l_neg[~mask] = -torch.inf

            #### v2 
            # mask = torch.eq(domain_label[0].contiguous().view(-1, 1), self.queue_labels.T).bool().to(device)  # (NxQ) = label of sen_q (Nx1) x labels of queue (Qx1).T
            # num_mask = torch.sum(~mask, dim=1)

            # sim_wih_other_domain = l_neg.clone().detach()
            # sim_wih_other_domain[mask] = -torch.inf  # then sample within the same domain would be -inf

            # sim_sorted, indices = torch.sort(sim_wih_other_domain, dim=1, descending=True)
            # dim_wise_len = (num_mask * self.sample_ratio).int()

            # for i in range(indices.shape[0]):
            #     mask[i, indices[i, : dim_wise_len[i]]] = True

            # l_neg[~mask] = -torch.inf

            #### v1
            # mask = torch.eq(domain_label[0].contiguous().view(-1, 1), self.queue_labels.T).bool().to(device)  # (NxQ) = label of sen_q (Nx1) x labels of queue (Qx1).T
            # l_neg[~mask] = -torch.inf

        
        feature = torch.concat([q, self.queue.clone().detach().T], dim=0)  # the overall features rather than the dot product of features.  

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # targets: positive key indicators
        targets = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, domain_label, gt, d_k, time_label)
        
        if self.DAL: # we do / T when calculating the SupCon
            logits_labels = torch.einsum('nc,ck->nk', [d_q, self.queue_dis.clone().detach()])
        else:
            logits_labels = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # if torch.isnan(q).any() or torch.isnan(k).any():
        #     pdb.set_trace()
        #     print("now")

        return logits, targets, logits_labels, hardest_related_info, similarity_across_domains, feature
    
    
    def supervised_CL(self, logits_labels=None, labels=None):
        if labels and self.label_type:
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
        logging.info(f"Start MoCo training for {self.args.epochs} epochs.")
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
                    else:
                        NotADirectoryError
                else:
                    domain_label = None
                    time_label = None
                with autocast(enabled=self.args.fp16_precision):
                    
                    output, target, logits_labels, hardest_related_info, similarity_across_domains, feature = self.model(sensor[0], sensor[1], 
                                                                                                                        domain_label=domain_label, 
                                                                                                                        gt=class_label, 
                                                                                                                        time_label=time_label)
                    if self.model.if_cross_entropy:
                        sup_loss = self.model.supervised_CL(logits_labels=feature, labels=domain_label)
                    else:
                        sup_loss = self.model.supervised_CL(logits_labels=logits_labels, labels=domain_label)

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
            # in current setting, it is not meaningful
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
            time_fw = AverageMeter('time_fw', ':6.5f')

            pred_batch = torch.empty(0).to(self.args.device)
            label_batch = torch.empty(0).to(self.args.device)

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
            
            start_time = time.time()
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
                end_time = time.time()

                f1 = f1_cal(logits, target, topk=(1,))
                acc = accuracy(logits, target, topk=(1,))
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

                if self.args.time_analysis:
                    time_fw.update(end_time - start_time, 1)
                    if n_iter_train > 5000:
                        break


            f1_batch = f1_score(label_batch.cpu().numpy(), pred_batch.cpu().numpy(), average='macro') * 100
            val_acc, val_f1 = MoCo_evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=val_loader)

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
            
            logging.debug(f"Epoch: {epoch_counter} Loss: {loss} acc: {acc_batch.avg: .3f}/{val_acc: .3f} f1: {f1_batch: .3f}/{val_f1: .3f}")
            if self.args.time_analysis and n_iter_train > 5000:
                logging.debug(f"time of one training round is {time_fw.avg}")
                print(f"time of one training round is {time_fw.avg}")
                break

        logging.info("Fine-tuning has finished.")
        logging.info(f"best eval f1 is {best_f1} at {best_epoch}.")

        print('best eval f1 is {} for {}'.format(best_f1, self.args.name))

    def test_performance(self, best_model_dir, test_loader):
        checkpoint = torch.load(best_model_dir, map_location="cpu")
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict)
        test_acc, test_f1 = MoCo_evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=test_loader, test=True)
        logging.info(f"test f1 is {test_f1}.")
        logging.info(f"test acc is {test_acc}.")

        print('test f1 is {} for {}'.format(test_f1, self.args.name))
        print('test acc is {} for {}'.format(test_acc, self.args.name))
    
    def test_performance_cross_dataset(self, best_model_dir, test_dataloader_for_all_datasets, datasets):
        checkpoint = torch.load(best_model_dir, map_location="cpu")
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        for i, test_loader in enumerate(test_dataloader_for_all_datasets):
            test_acc, test_f1 = MoCo_evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=test_loader)

            logging.info(f"test f1 is {test_f1} for {datasets[i]}.")
            logging.info(f"test acc is {test_acc} for {datasets[i]}.")

            print('test f1 is {} for {}'.format(test_f1, datasets[i]))
            print('test acc is {} for {}'.format(test_acc, datasets[i]))

    
    def transfer_train_DAL(self, tune_loader, val_loader, train_loader):
        """
        train loader is for domain adversial learning
        we only consider fine-tune + DAL. Linear Evaluation is not considered for this type of learning.

        Fix one train one, the performance is not satisfying. 
        We turn to another design. 
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

    
    def transfer_train_DAL_v2(self, tune_loader, val_loader, train_loader):
        """
        train loader is for domain adversial learning
        we only consider fine-tune + DAL. Linear Evaluation is not considered for this type of learning.

        Fix one train one, the performance is not satisfying. 
        We turn to another design. 
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
            
            data_loader = zip(tune_loader, train_loader)

            for (sensor, target), (sensor_domain, target_domain) in data_loader:

                sensor= sensor.to(self.args.device)
                sensor_domain = sensor_domain.to(self.args.device)
                
                target = target[:, 0].to(self.args.device)
                target_domain = target_domain[:, 1].to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    h = self.model.encoder(torch.concat([sensor, sensor_domain], dim=0))
                    logits = self.model.classifier(h[:sensor.shape[0]])
                    loss_class = self.criterion(logits, target)  # fine-tune for HAR

                    logits_domain = self.model.discriminator(h[sensor.shape[0]:])
                    loss_domain = self.criterion(logits_domain, target_domain)

                    loss = loss_class + self.args.dlr * loss_domain

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
                    self.writer.add_scalar('loss_class', loss_class, global_step=n_iter_train)
                    self.writer.add_scalar('loss_domain', loss_domain, global_step=n_iter_train)

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
            logging.debug(f"Epoch: {epoch_counter} Loss: {loss} loss_class: {loss_class} loss_domain: {loss_domain} acc: {acc_batch.avg: .3f}/{val_acc: .3f} f1: {f1_batch.avg: .3f}/{val_f1: .3f}")

        logging.info("Fine-tuning has finished.")
        logging.info(f"best eval f1 is {best_f1} at {best_epoch}.")

        print('best eval f1 is {} for {}'.format(best_f1, self.args.name))
    
    def transfer_train_ewc(self, tune_loader, val_loader, fisher):
        assert self.args.if_fine_tune is True

        self.fisher = fisher
        self.mean = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        
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
            time_fw = AverageMeter('time_fw', ':6.5f')

            pred_batch = torch.empty(0).to(self.args.device)
            label_batch = torch.empty(0).to(self.args.device)

            # f1_batch = AverageMeter('f1_batch', ':6.2f')
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
                
            start_time = time.time()
            for sensor, target in tune_loader:
                sensor = sensor.to(self.args.device)
                target = target[:, 0].to(self.args.device)
                label_batch = torch.cat((label_batch, target))

                with autocast(enabled=self.args.fp16_precision):
                    logits, _ = self.model(sensor)
                    loss_clf = self.criterion(logits, target)

                    loss_ewc = self.compute_ewc()

                    loss = loss_clf + self.args.ewc_lambda * loss_ewc

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                end_time = time.time()
                    
                acc = accuracy(logits, target, topk=(1,))
                _, pred = logits.topk(1, 1, True, True)
                pred_batch = torch.cat((pred_batch, pred.reshape(-1)))
                
                f1 = f1_cal(logits, target, topk=(1,))
                acc_batch.update(acc, sensor.size(0))
                # f1_batch.update(f1, sensor.size(0))
                if n_iter_train % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter_train)
                    self.writer.add_scalar('loss_clf', loss_clf, global_step=n_iter_train)
                    self.writer.add_scalar('loss_ewc', loss_ewc, global_step=n_iter_train)
                    self.writer.add_scalar('acc', acc, global_step=n_iter_train)
                    self.writer.add_scalar('f1', f1, global_step=n_iter_train)
                    self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], global_step=n_iter_train)

                n_iter_train += 1

                if self.args.time_analysis:
                    time_fw.update(end_time - start_time, 1)
                    if n_iter_train > 5000:
                        break

            val_acc, val_f1 = MoCo_evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=val_loader)

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

            f1_batch = f1_score(label_batch.cpu().numpy(), pred_batch.cpu().numpy(), average='macro') * 100
            logging.debug(f"Epoch: {epoch_counter} Loss: {loss} Loss_EWC: {loss_ewc} acc: {acc_batch.avg: .3f}/{val_acc: .3f} f1: {f1_batch: .3f}/{val_f1: .3f}")

            if self.args.time_analysis and n_iter_train > 5000:
                logging.debug(f"time of one training round is {time_fw.avg}")
                print(f"time of one training round is {time_fw.avg}")
                break

        logging.info("Fine-tuning has finished.")
        logging.info(f"best eval f1 is {best_f1} at {best_epoch}.")

        print('best eval f1 is {} for {}'.format(best_f1, self.args.name))

    def compute_ewc(self):
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher.keys():
                loss += (
                    torch.sum(
                        (self.fisher[n])
                        * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                    )
                    / 2
                )
        return loss
    
    def transfer_train_SCL(self, tune_loader, val_loader, train_loader):
        """
        train loader is for supervised contrastive learning
        we consider fine-tune + SCL. Linear Evaluation is not considered for this type of learning.
        """

        assert self.args.if_fine_tune is True
        
        sup_loss = SupConLoss(device=self.args.device)

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
            self.model.train()
            
            data_loader = zip(tune_loader, train_loader)

            for (sensor, target), (sensor_domain, target_domain) in data_loader:

                sensor= sensor.to(self.args.device)
                sensor_domain = sensor_domain.to(self.args.device)
                
                target = target[:, 0].to(self.args.device)
                target_domain = target_domain[:, 1].to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    h = self.model.encoder(torch.concat([sensor, sensor_domain], dim=0))
                    logits = self.model.classifier(h[:sensor.shape[0]])
                    loss_class = self.criterion(logits, target)  # fine-tune for HAR

                    logits_domain = self.model.projector(h[sensor.shape[0]:])
                    logits_domain = F.normalize(logits_domain, dim=1)
                    logits_domain = torch.einsum('nc,kc->nk', [logits_domain, logits_domain])
                    logits_domain = logits_domain/self.args.tem_labels

                    loss_domain = sup_loss.transfer_calculate(logits=logits_domain, labels=target_domain)

                    loss = loss_class - self.args.cl_slr * loss_domain

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
                    self.writer.add_scalar('loss_class', loss_class, global_step=n_iter_train)
                    self.writer.add_scalar('loss_domain', loss_domain, global_step=n_iter_train)

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
            logging.debug(f"Epoch: {epoch_counter} Loss: {loss} loss_class: {loss_class} loss_domain: {loss_domain} acc: {acc_batch.avg: .3f}/{val_acc: .3f} f1: {f1_batch.avg: .3f}/{val_f1: .3f}")

        logging.info("Fine-tuning has finished.")
        logging.info(f"best eval f1 is {best_f1} at {best_epoch}.")

        print('best eval f1 is {} for {}'.format(best_f1, self.args.name))
        return

    def transfer_train_ewc_mixup(self, tune_loader, val_loader, fisher):
        assert self.args.if_fine_tune is True

        self.fisher = fisher
        self.mean = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        
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
            # acc_batch = AverageMeter('acc_batch', ':6.2f')

            pred_batch = torch.empty(0).to(self.args.device)
            # label_batch = torch.empty(0).to(self.args.device)
            label_a_batch = torch.empty(0).to(self.args.device)
            label_b_batch = torch.empty(0).to(self.args.device)

            # f1_batch = AverageMeter('f1_batch', ':6.2f')
            if self.args.if_fine_tune:
                self.model.train()
            else:  
                self.model.eval()
                self.model.classifier.train()
            
            lam = np.random.beta(1.0, 1.0)
                
            for sensor, target in tune_loader:
                sensor, targets_a, targets_b = self.mixup_data(sensor, target[:, 0], lam)

                sensor = sensor.to(self.args.device)
                targets_a, targets_b = targets_a.to(self.args.device), targets_b.to(self.args.device)

                label_a_batch, label_b_batch = torch.cat((label_a_batch, targets_a)), torch.cat((label_b_batch, targets_b))

                with autocast(enabled=self.args.fp16_precision):
                    logits, _ = self.model(sensor)
                    loss_clf = self.mixup_criterion(logits, targets_a, targets_b, lam)
                    loss_ewc = self.compute_ewc()

                    loss = loss_clf + self.args.ewc_lambda * loss_ewc

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                _, pred = logits.topk(1, 1, True, True)
                pred_batch = torch.cat((pred_batch, pred.reshape(-1)))
                
                acc = mixup_accuracy(pred, targets_a, targets_b, lam)
                f1 = mixup_f1(pred, targets_a, targets_b, lam)

                if n_iter_train % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter_train)
                    self.writer.add_scalar('loss_clf', loss_clf, global_step=n_iter_train)
                    self.writer.add_scalar('loss_ewc', loss_ewc, global_step=n_iter_train)
                    self.writer.add_scalar('acc', acc, global_step=n_iter_train)
                    self.writer.add_scalar('f1', f1, global_step=n_iter_train)
                    self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], global_step=n_iter_train)

                n_iter_train += 1

            f1_batch = mixup_f1(pred_batch, label_a_batch, label_b_batch, lam)
            acc_batch = mixup_accuracy(pred_batch, label_a_batch, label_b_batch, lam)

            val_acc, val_f1 = MoCo_evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=val_loader)

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

            logging.debug(f"Epoch: {epoch_counter} Loss: {loss} acc: {acc_batch: .3f}/{val_acc: .3f} f1: {f1_batch: .3f}/{val_f1: .3f}")

        logging.info("Fine-tuning has finished.")
        logging.info(f"best eval f1 is {best_f1} at {best_epoch}.")

        print('best eval f1 is {} for {}'.format(best_f1, self.args.name))

    def mixup_data(self, x, y, lam):
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.args.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b

    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    
def mixup_accuracy(predicted, y_a, y_b, lam):
    with torch.no_grad():
        predicted = predicted.reshape(-1)
        correct = (lam * predicted.eq(y_a).cpu().sum().float()
                        + (1 - lam) * predicted.eq(y_b).cpu().sum().float())
    return 100 * correct / y_a.shape[0]

def mixup_f1(pred, y_a, y_b, lam):
    with torch.no_grad():
        f1_a = f1_score(y_a.cpu().numpy(), pred.cpu().numpy(), average='macro') * 100
        f1_b = f1_score(y_b.cpu().numpy(), pred.cpu().numpy(), average='macro') * 100
    return lam * f1_a + (1 - lam) * f1_b