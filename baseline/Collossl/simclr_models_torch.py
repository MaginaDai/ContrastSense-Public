import torch
import torch.nn as nn
import torch.nn.functional as F


class TPN_encoder(nn.Module):
    def __init__(self):
        super(TPN_encoder, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.convA = nn.Conv2d(6, 32, kernel_size=(24, 1), stride=1)
        self.convB = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=1)
        self.convC = nn.Conv2d(64, 96, kernel_size=(8, 1), stride=1)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.permute(0, 3, 2, 1)
        else:
            x = x.unsqueeze(1).permute(0, 3, 2, 1)
        h = self.dropout(self.relu(self.convA(x)))
        h = self.dropout(self.relu(self.convB(h)))
        h = self.dropout(self.relu(self.convC(h)))
        h = h.reshape(h.shape[0], h.shape[1], -1)
        h = F.max_pool1d(h, kernel_size=int(h.size()[-1])) 
        return h