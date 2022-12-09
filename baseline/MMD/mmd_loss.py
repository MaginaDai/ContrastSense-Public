import torch
import torch.nn as nn


class MMD_loss(nn.Module):
    def __init__(self):
        super(MMD_loss, self).__init__()
        self.fix_sigma = torch.tensor([0.0001,0.001,0.01,0.1,1,5,10,15,20,25,30,100])
        

    def forward(self, feature, domain):
        domain = domain.unsqueeze(1)
        mask = torch.eq(domain, domain.T).float()
        neg_mask = (~torch.eq(domain, domain.T)).float()

        total0 = feature.unsqueeze(0).expand(int(feature.size(0)), int(feature.size(0)), int(feature.size(1)))
        total1 = feature.unsqueeze(1).expand(int(feature.size(0)), int(feature.size(0)), int(feature.size(1)))

        L2_distance = ((total0-total1)**2).sum(2)
        kernel_val = [torch.exp(-L2_distance / 2 * sigma) for sigma in self.fix_sigma]
        k = sum(kernel_val)
        
        Ex = torch.sum(mask * k) / torch.sum(mask)
        Ey = torch.sum(neg_mask * k) / torch.sum(neg_mask)

        loss = Ex - Ey
        
        return loss



if __name__ == '__main__':
    source = torch.randn([32, 100])
    target = torch.randn([64, 100])
    feature = torch.cat([source, target], dim=0)
    domain_x = torch.ones([32, 1])
    domain_y = torch.zeros([64, 1])
    domain = torch.cat([domain_x, domain_y], dim=0)

    mmd_loss = MMD_loss()
    
    loss = mmd_loss(feature, domain)