import torch
import torch.nn as nn


class CLISA_model(nn.Module):
    def __init__(self, transfer=False, num_class=4):
        super(CLISA_model, self).__init__()
        self.transfer = transfer

        self.encoder = CLISA_encoder()

        if self.transfer:
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_class)
            )
        else:
            self.projector = CLISA_projector()

    
    def forward(self, x):
        h = self.encoder(x)
        if self.transfer:
            z = self.classifier(h)
        else:
            z = self.projector(h)
        return z
    

class CLISA_encoder(nn.Module):
    def __init__(self):
        super(CLISA_encoder, self).__init__()
        self.Spa_Conv = torch.nn.Conv1d(62, 16, kernel_size=1)
        self.Temp_Conv = torch.nn.Conv1d(16, 16, kernel_size=48)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x=self.Spa_Conv(x)
        x=self.Temp_Conv(x)
        return x
    

class CLISA_projector(nn.Module):
    def __init__(self):
        super(CLISA_projector, self).__init__()
        self.pool = torch.nn.AvgPool1d(24)
        self.Spa_Conv = torch.nn.Conv1d(16, 32, kernel_size=1)
        self.Temp_Conv = torch.nn.Conv1d(32, 64, kernel_size=4)

    def forward(self, x):
        x=self.pool(x)
        x=self.Spa_Conv(x)
        x=self.Temp_Conv(x)
        return x.reshape(x.shape[0], -1)