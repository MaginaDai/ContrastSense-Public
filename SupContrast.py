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
    def __init__(self, device, if_cross_entropy=False):
        super(SupConLoss, self).__init__()
        self.if_cross_entropy = if_cross_entropy
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def forward(self, logits, labels=None, queue_labels=None):
        """Compute supervised loss for model."""
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
        # mask_sum = mask.sum(1)
        # mask_zero = mask_sum != 0
        # loss = -(mean_log_prob_pos * mask_zero).sum()/ mask_zero.sum()
        # loss
        loss = - mean_log_prob_pos.mean()
        if torch.isnan(loss).any():
            print(logits)
            print(mask.sum(1))
            print(mean_log_prob_pos)
            pdb.set_trace()
        return loss
    
    def transfer_calculate(self, logits, labels):
        batch_size = logits.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        
        mask = torch.eq(labels, labels.T).float().to(self.device)  # (NxQ) = label of sen_q (Nx1) x labels of queue (Qx1).T

        mask = mask - torch.eye(batch_size).to(self.device)  # now is self-comparison so we need to tune it.

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # Only one data point from this domain. SCL requir at least two data form the same domain. So we mask the corresponding value in this case.
        non_zero_pos = torch.nonzero(mask.sum(1))[:, 0]
        log_prob_non_zero = log_prob[non_zero_pos, :]
        mask_non_zero = mask[non_zero_pos, :]
        
        mean_log_prob_pos = (mask_non_zero * log_prob_non_zero).sum(1) / mask_non_zero.sum(1)
        loss = - mean_log_prob_pos.mean()

        if torch.isnan(loss).any():
            print(logits)
            print(mask.sum(1))
            print(mean_log_prob_pos)
            pdb.set_trace()
        
        return loss
