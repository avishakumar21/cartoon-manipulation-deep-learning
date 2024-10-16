import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.nn.functional import normalize
from models.networks.base_network import BaseNetwork
from models.networks.utils import dis_conv

class DeepFillCDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super(DeepFillCDiscriminator, self).__init__()
        cnum=64
        self.conv1 = nn.utils.spectral_norm(dis_conv(4, cnum))
        self.conv2 = nn.utils.spectral_norm(dis_conv(cnum, cnum*2))
        self.conv3 = nn.utils.spectral_norm(dis_conv(cnum*2, cnum*4))
        self.conv4 = nn.utils.spectral_norm(dis_conv(cnum*4, cnum*4))
        self.conv5 = nn.utils.spectral_norm(dis_conv(cnum*4, cnum*4))
        self.conv6 = nn.utils.spectral_norm(dis_conv(cnum*4, cnum*4))

    def forward(self, x, guide, cc=None):
        bsize, ch, height, width = x.shape
        x = torch.cat([x, guide], 1)
        ## [image, edge]
        ## edge=1 as an extra channel to use tf pretrained model
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

if __name__ == "__main__":
    pass
