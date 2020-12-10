import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

coords_size = (248, 2)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Generator(nn.Module):
    def __init__(self, latent_dim):
        
        super(Generator, self).__init__()

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, noise, labels):
        # 1. fully_connected
        gen_input = torch.cat((labels, noise), -1)

        return ub

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.depth = 64
        self.kernel_size = (4,2)

        self.conv1 = nn.Conv2d(2, self.depth//2, kernel_size=(2,2), stride=(2,2), padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        # self.dropout1 = nn.Dropout(0.4)

        self.conv2 = nn.Conv2d(self.depth//2, self.depth*1, kernel_size=(2,2), stride=(2,2), padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout2 = nn.Dropout(0.4)

        self.conv3 = nn.Conv2d(self.depth*1, self.depth*2, kernel_size=(2,2), stride=(2,2), padding=1)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout3 = nn.Dropout(0.4)

        self.conv4 = nn.Conv2d(self.depth*2, self.depth*4, kernel_size=(4,2), stride=(2,2), padding=1)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout4 = nn.Dropout(0.4)
        
        self.conv5 = nn.Conv2d(self.depth*4, self.depth*8, kernel_size=(4,2), stride=(2,2), padding=1)
        self.lrelu5 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout5 = nn.Dropout(0.4)

        self.conv6 = nn.Conv2d(self.depth*8, self.depth*16, kernel_size=(4,2), stride=(2,2), padding=1)
        self.lrelu6 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout6 = nn.Dropout(0.4)
        self.conv7 = nn.Conv2d(self.depth*16, self.depth*32, kernel_size=(4,2), stride=(2,2), padding=1)
        self.lrelu7 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout7 = nn.Dropout(0.4)
        
        self.flatten = nn.Flatten()
        self.activation = nn.Sigmoid()

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    def forward(self, coords, labels):
        d_in = torch.cat([coords, labels], 1)
        d = self.lrelu1(self.conv1(d_in))
        d = self.dropout2(self.lrelu2(self.conv2(d)))
        d = self.dropout3(self.lrelu3(self.conv3(d)))
        d = self.dropout4(self.lrelu4(self.conv4(d)))
        d = self.dropout5(self.lrelu5(self.conv5(d)))
        d = self.dropout6(self.lrelu6(self.conv6(d)))
        d = self.dropout7(self.lrelu7(self.conv7(d)))
        d = self.flatten(d)
        d = nn.Linear(d.shape[1], 1)(d)
        validity = self.activation(d)
        return validity
