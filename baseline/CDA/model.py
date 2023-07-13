## The STCN-GR model basically follow the design in the following model, 2s-AGCN
## We develop base on their code

##  @inproceedings{2sagcn2019cvpr,  
#       title     = {Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition},  
#       author    = {Lei Shi and Yifan Zhang and Jian Cheng and Hanqing Lu},  
#       booktitle = {CVPR},  
#       year      = {2019},  
# }

## Github Page https://github.com/lshiwjx/2s-AGCN


import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(unit_tcn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(9, 1), stride=1, padding=(4, 0)) 
        # we add padding to the time axis to keep the len the same for residual

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x
    

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_subset=1):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels
        self.inter_c = inter_channels

        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        self.num_subset = num_subset

        self.conv = nn.Conv2d(in_channels, out_channels, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

    def forward(self, x):

        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = self.conv(torch.matmul(x, A))
        y = self.bn(y)
        y += self.down(x)

        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels)
        self.relu = nn.ReLU()

        self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.down(x)
        return self.relu(x)
    

class STCN(nn.Module):
    def __init__(self, num_class=None, transfer=False):
        super(STCN, self).__init__()
        A = np.ones([8, 8])

        self.l1 = TCN_GCN_unit(1, 4, A)
        self.l2 = TCN_GCN_unit(4, 8, A)
        self.l3 = TCN_GCN_unit(8, 8, A)
        self.l4 = TCN_GCN_unit(8, num_class, A)

        self.transfer = transfer
        self.GAP = nn.AdaptiveAvgPool2d(1)
        if self.transfer:
            self.Classifier = nn.Sequential(
                nn.Linear(num_class, num_class)
            )
        else:
            self.sim_head = nn.Sequential(
                nn.Linear(num_class, 512),
                nn.Linear(512, 128)
            )

    def forward(self, x):

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.GAP(x)
        x = x.reshape(x.shape[0], -1)

        if self.transfer:
            x = self.Classifier(x)
        else:
            x = self.sim_head(x)

        return x