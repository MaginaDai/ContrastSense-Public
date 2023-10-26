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

    def forward(self, sen_q, sen_k, domain_label):
        """
        Input:
            sen_q: a batch of query sensors data
            sen_k: a batch of key sensors data
            domain_label: the label for nuisance suppression
        Output:
            logits, targets
        """

        q, d_q = self.encoder_q(sen_q)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k, d_k = self.encoder_k(sen_k)  # keys: NxC
            k = F.normalize(k, dim=1)
        
        features = torch.cat([q, k], dim=0)

        logits, targets = self.info_nce_loss(features)
        
        return logits, targets

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)


        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        logits = logits / self.T
        return logits, labels