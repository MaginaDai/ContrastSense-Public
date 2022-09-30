import torch
from simclr import MyNet
from torchsummary import summary

if __name__ == '__main__':
    x = torch.randn((256, 1, 100, 6))
    model = MyNet(transfer=False, out_dim=32)
    model.to(torch.device('cuda'))
    # y = model(x)
    summary(model, input_size=(1, 100, 6))