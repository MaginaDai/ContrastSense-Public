import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class TPN_encoder(nn.Module):
    def __init__(self):
        super(TPN_encoder, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.convA = nn.Conv1d(6, 32, kernel_size=(24, 1), stride=1)
        self.convB = nn.Conv1d(32, 64, kernel_size=(16, 1), stride=1)
        self.convC = nn.Conv1d(64, 96, kernel_size=(8, 1), stride=1)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        h = self.dropout(self.relu(self.convA(x)))
        h = self.dropout(self.relu(self.convB(h)))
        h = self.dropout(self.relu(self.convC(h)))
        h = h.reshape(h.shape[0], h.shape[1], -1)
        h = F.max_pool1d(h, kernel_size=h.size()[-1]) 
        return h
    

class Cluster_Projector(nn.Module):
    def __init__(self):
        super(Cluster_Projector, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(96, 96)
        self.linear2 = nn.Linear(96, 96)
        self.linear3 = nn.Linear(96, 96)
        
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class CL_Projector(nn.Module):
    def __init__(self):
        super(CL_Projector, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(96, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 50)
        self.conv1_acc_BN =torch.nn.BatchNorm2d(num_features=256, momentum=0.9, affine=False)
        self.conv1_acc_BN =torch.nn.BatchNorm2d(num_features=128, momentum=0.9, affine=False)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Cluster_Classifier(nn.Module):
    def __init__(self, classes):
        super(Cluster_Classifier, self).__init__()
        self.linear1 = nn.Linear(96, 96)
        self.linear2 = nn.Linear(96, 48)
        self.linear3 = nn.Linear(48, classes)
        
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        return x


class CL_Classifier(nn.Module):
    def __init__(self, classes):
        super(CL_Classifier, self).__init__()
        self.linear1 = nn.Linear(96, 1024)
        self.linear2 = nn.Linear(1024, classes)
        self.relu = nn.ReLU()
        
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, method):
        super(Encoder, self).__init__()
        self.TPN = TPN_encoder()
        if method == 'CL':
            self.Head = CL_Projector()
        elif method == 'Cluster':
            self.Head = Cluster_Projector()
        else:
            raise TypeError
    
    def forward(self, x):
        h = self.TPN(x)
        z = self.Head(h)
        return z


class Transfer_Coder(nn.Module):
    def __init__(self, classes, method):
        super(Transfer_Coder, self).__init__()
        self.TPN = TPN_encoder()
        if method == 'CL':
            self.Classifier = CL_Classifier(classes)
        elif method == 'Cluster':
            self.Classifier = Cluster_Classifier(classes)
        else:
            raise TypeError
    
    def forward(self, x):
        h = self.TPN(x)
        z = self.Classifier(h)
        return z
