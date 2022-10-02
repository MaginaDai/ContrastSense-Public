"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function
from cmath import nan
import pdb

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, device, temperature=0.07, base_temperature=0.07, if_cross_entropy=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.if_cross_entropy = if_cross_entropy
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def forward(self, logits, labels=None, queue_labels=None):
        """Compute supervised loss for model. """
        if self.if_cross_entropy:
            target = torch.concat([labels, queue_labels.squeeze()])
            loss = self.criterion(logits, target)  # use cross entropy for loss calculation.
            return loss
        
        batch_size = logits.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        mask = torch.eq(labels, queue_labels.T).float().to(self.device)  # (NxQ) = label of sen_q (Nx1) x labels of queue (Qx1).T

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        if torch.isnan(loss).any():
            pdb.set_trace()
        return loss
