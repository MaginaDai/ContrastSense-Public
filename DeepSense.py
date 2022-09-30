import pdb
from traceback import print_tb
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

CONV_LEN = 3
CONV_LEN_INTE = 3#4
CONV_LEN_LAST = 3#5
CONV_NUM = 64
CONV_MERGE_LEN = 8
CONV_MERGE_LEN2 = 6
CONV_MERGE_LEN3 = 4
CONV_NUM2 = 64

class DeepSense_encoder(nn.Module):

    def __init__(self, transfer=False, out_dim=512, classes=6, dims=64):
        super(DeepSense_encoder, self).__init__()
        self.transfer = transfer

        self.dropout = torch.nn.Dropout(p=0.1)
        self.relu = torch.nn.ReLU()

        # (N, Channel=1, H, W)
        self.conv1_acc = torch.nn.Conv2d(1, dims, kernel_size=(2*3*CONV_LEN, 1), stride=[1, 2*3])
        self.conv1_acc_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)
        
        self.conv2_acc = torch.nn.Conv2d(dims, dims, kernel_size=(CONV_LEN_INTE, 1), stride=[1, 1])
        self.conv2_acc_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)

        self.conv3_acc = torch.nn.Conv2d(dims, dims, kernel_size=(CONV_LEN_LAST, 1), stride=[1, 1])
        self.conv3_acc_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)

        self.conv1_gyro = torch.nn.Conv2d(dims, dims, kernel_size=(2*3*CONV_LEN, 1), stride=[1, 2*3])
        self.conv1_gyro_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)
        
        self.conv2_gyro = torch.nn.Conv2d(dims, dims, kernel_size=(CONV_LEN_INTE, 1), stride=[1, 1])
        self.conv2_gyro_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)

        self.conv3_gyro = torch.nn.Conv2d(dims, dims, kernel_size=(CONV_LEN_LAST, 1), stride=[1, 1])
        self.conv3_gyro_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)

        self.conv1_sensor = torch.nn.Conv2d(dims, dims, kernel_size=(CONV_MERGE_LEN, 1), stride=[1, 1])
        self.conv1_sensor_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)
        
        self.conv2_sensor = torch.nn.Conv2d(dims, dims, kernel_size=(CONV_MERGE_LEN2, 1), stride=[1, 1])
        self.conv2_sensor_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)

        self.conv3_sensor = torch.nn.Conv2d(dims, dims, kernel_size=(CONV_MERGE_LEN3, 1), stride=[1, 1])
        self.conv3_sensor_BN =torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)
        
        self.gru1 = torch.nn.GRU(dims, int(dims/2), num_layers=1, batch_first=True, bidirectional=True)
        self.gru2 = torch.nn.GRU(dims, int(dims/2), num_layers=1, batch_first=True, bidirectional=True)
        if transfer:
            self.gru3 = torch.nn.GRU(dims, int(dims/8), num_layers=1, batch_first=True, bidirectional=True)
            self.classifier = nn.Sequential(nn.Linear(in_features=1344, out_features=128),
                                            nn.ReLU(),
                                            nn.Linear(in_features=128, out_features=classes))
        else:
            self.projector = nn.Sequential(nn.Linear(in_features=5376, out_features=1024),
                                            nn.ReLU(),
                                            nn.Linear(in_features=1024, out_features=out_dim))
        
    
    def forward(self, x):
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        if self.transfer:
            self.gru3.flatten_parameters()

        # data shape: (BATCH_SIZE, CHANNEL=1, Timestamps, FEATURE_DIM)
        h_acc = self.conv1_acc_BN(self.conv1_acc(x[:, :, :, 0:3]))
        h_acc = self.dropout(self.relu(h_acc))

        h_acc = self.conv2_acc_BN(self.conv2_acc(h_acc))
        h_acc = self.dropout(self.relu(h_acc))

        h_acc = self.conv3_acc_BN(self.conv3_acc(h_acc))
        h_acc = self.relu(h_acc)
        h_acc_shape = h_acc.shape
        h_acc_out = h_acc.reshape(h_acc_shape[0], h_acc_shape[1], -1, 1)
        
        h_gyro = self.conv1_acc_BN(self.conv1_acc(x[:, :, :, 3:]))
        h_gyro = self.dropout(self.relu(h_gyro))

        h_gyro = self.conv2_acc_BN(self.conv2_acc(h_gyro))
        h_gyro = self.dropout(self.relu(h_gyro))

        h_gyro = self.conv3_acc_BN(self.conv3_acc(h_gyro))
        h_gyro = self.relu(h_gyro)
        h_gyro_shape = h_gyro.shape
        h_gyro_out = h_gyro.reshape(h_gyro_shape[0], h_gyro_shape[1], -1, 1)
        
        
        # reshape to (BATCH_SIZE, Channels, FEATURE_DIM, 2)
        h = torch.cat((h_acc_out, h_gyro_out), dim=3)
        h = self.dropout(h)

        h = self.conv1_sensor_BN(self.conv1_sensor(h))
        h = self.dropout(self.relu(h))

        h = self.conv2_sensor_BN(self.conv2_sensor(h))
        h = self.dropout(self.relu(h))

        h = self.conv3_sensor_BN(self.conv3_sensor(h))
        h = self.relu(h)

        # reshape to (BATCH_SIZE, FEATURE_DIM, 2)
        h = h.flatten(start_dim=2)
        h_shape = h.shape
        h = h.reshape(h.shape[0], h_shape[2], -1)
        h, _ = self.gru1(h)
        h = self.dropout(h)
        h, _ = self.gru2(h)
        h = self.dropout(h)

        if self.transfer:
            h, _ = self.gru3(h)
            h = self.dropout(h)
            h = h.flatten(start_dim=1)
            h = self.classifier(h)
        else:
            h = h.flatten(start_dim=1)
            h = self.projector(h)
        return h
        







