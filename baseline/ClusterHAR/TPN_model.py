import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class TPN_encoder(nn.Module):
    def __init__(self):
        super(TPN_encoder, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.convA = nn.Conv1d(1, 32, kernel_size=(24, 1), stride=1)
        self.convB = nn.Conv1d(32, 64, kernel_size=(16, 1), stride=1)
        self.convC = nn.Conv1d(64, 96, kernel_size=(8, 1), stride=1)
        nn.MaxPool1d

    def forward(self, x):
        h = self.dropout(self.relu(self.convA(x)))
        h = self.dropout(self.relu(self.convB(h)))
        h = self.dropout(self.relu(self.convC(h)))
        h = h.permute(0, 2, 1, 3)
        h = h.reshape(h.shape[0], h.shape[1], -1)
        h = F.max_pool1d(h, kernel_size=h.size()[-1])  # apply to the feature dims I think
        return h
    

class Projector(nn.Module):
    def __init__(self):
        super(Projector, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(155, 96)
        self.linear2 = nn.Linear(96, 96)
        self.linear3 = nn.Linear(96, 96)
        
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Classifier(nn.Module):
    def __init__(self, classes):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(155, 96)
        self.linear2 = nn.Linear(96, 48)
        self.linear3 = nn.Linear(48, classes)
        
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.TPN = TPN_encoder()
        self.Head = Projector()
    
    def forward(self, x):
        h = self.TPN(x)
        z = self.Head(h)
        return z


class Transfer_Coder(nn.Module):
    def __init__(self, classes):
        super(Transfer_Coder, self).__init__()
        self.TPN = TPN_encoder()
        self.Classifier = Classifier(classes)
    
    def forward(self, x):
        h = self.TPN(x)
        z = self.Classifier(h)
        return z
