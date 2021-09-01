import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from models_utils import BasicBlockTimeBest, NetworkBlockOld
    
    
class WideResNet128Best1380(nn.Module):
    def __init__(self, mel_bins, depth=28, widen=10, t=1):
        super(WideResNet128Best1380, self).__init__()

        emb_size = 128 * widen
        nChannels = [8 * widen, 16 * widen, 32 * widen, 64 * widen, 128 * widen]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlockTimeBest
        
        if t == 1:
            t1 = 2
            t2 = 2
            t3 = 2
        elif t == 2:
            t1 = 4
            t2 = 2
            t3 = 2
        elif t == 4:
            t1 = 4
            t2 = 4
            t3 = 2
        elif t == 8:
            t1 = 4
            t2 = 4
            t3 = 4
        
        strides = [(t1, 2), (t2, 2), (t3, 2), (2, 2)]
        
        self.block1 = NetworkBlockOld(n, 1, nChannels[0], block, 1, 0)
        self.block2 = NetworkBlockOld(n, nChannels[0], nChannels[1], block, (t1, 2), 0)
        self.block3 = NetworkBlockOld(n, nChannels[1], nChannels[2], block, (t2, 2), 0)
        self.block4 = NetworkBlockOld(n, nChannels[2], nChannels[3], block, (t3, 2), 0)
        self.block5 = NetworkBlockOld(n, nChannels[3], nChannels[4], block, (2, 2), 0)

        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.bn2 = nn.BatchNorm2d(nChannels[0])
        self.bn3 = nn.BatchNorm2d(nChannels[1])
        self.bn4 = nn.BatchNorm2d(nChannels[2])
        self.bn5 = nn.BatchNorm2d(nChannels[3])
        self.bn6 = nn.BatchNorm2d(nChannels[4])
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.fc = nn.Linear(nChannels[-1], emb_size, bias=True)
        
        self.channels_last = nChannels[-1]
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
  

    def forward(self, x):
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3) 
        x = self.block1(x)
        x = self.relu(self.bn2(x))
        x = self.block2(x)
        x = self.relu(self.bn3(x))
        x = self.block3(x)
        x = self.relu(self.bn4(x))
        x = self.block4(x)
        x = self.relu(self.bn5(x))
        x = self.block5(x)
        x = self.relu(self.bn6(x))
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        res = x1 + x2
        res = self.relu(self.fc(res))
        
        return res
    
    
    
