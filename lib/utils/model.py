import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models

def count_parameters(model, verbose=True):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        if verbose: print([name, param])
        total_params+=param
    if verbose: print(f"Total Trainable Params: {total_params}")
    return total_params

class ResidualCNN(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.cnn1 = nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1)
        self.cnn2 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.is_shortcut = in_channels!=out_channels or stride!=1
        if self.is_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = self.shortcut(x) if self.is_shortcut else x
        x = self.cnn1(x)
        x = self.cnn2(x)
        x += residual
        x = self.bn(x)
        x = self.act(x)
        return x

class BNLRCNN(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, xs):
        return self.block(xs)

class Model_kp_ln(nn.Module):
    
    def __init__(self, n_class=4, has_attention=True):
        super().__init__()
        self.has_attention = has_attention
        self.n_class = n_class
        
        if self.has_attention:
            self.net_global_feature = nn.Sequential(
                BNLRCNN(3, 32, 7, 2, 3),      # 2x
                BNLRCNN(32, 64, 5, 2, 2),     # 4x
                ResidualCNN(64, 128, 2),       # 8x
                ResidualCNN(128, 256, 2),      # 16x
                ResidualCNN(256, 512, 2),     # 32x
                ResidualCNN(512, 128, 2),      # 64x
                nn.AdaptiveAvgPool2d(1),
                nn.Sigmoid(),
            )
        
        self.down128 = nn.Sequential(
            BNLRCNN(3, 32, 7, 2, 3),
            BNLRCNN(32, 64, 5, 2, 2),
        ) # 4x
        self.down64 = nn.Sequential(
            ResidualCNN(64, 128, 2),
            ResidualCNN(128, 128, 1),
        ) # 8x
        self.down32 = nn.Sequential(
            ResidualCNN(128, 256, 2),
            ResidualCNN(256, 256, 1),
        ) # 16x
        self.down16 = nn.Sequential(
            ResidualCNN(256, 512, 2),
            ResidualCNN(512, 512, 1),
        ) # 32x
        self.up32 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0),
            ResidualCNN(256, 256, 1),
        ) # 16x
        self.up64 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0),
            ResidualCNN(128, 128, 1),
        ) # 8x
        self.up128 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0),
            ResidualCNN(64, 64, 1),
        ) # 4x
        
        self.up512_kp = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 4, stride=4, padding=0),
            nn.Conv2d(128, 1, 3, stride=1, padding=1),
        ) # 1x
        self.up512_ln = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 4, stride=4, padding=0),
            nn.Conv2d(128, self.n_class, 3, stride=1, padding=1),
        ) # 1x
    
    def forward(self, xs):
        if self.has_attention:
            global_feat = self.net_global_feature(xs)
        
        d128 = self.down128(xs)
        d64  = self.down64(d128)
        d32  = self.down32(d64)
        d16  = self.down16(d32)
        u32  = self.up32(d16)+d32
        if self.has_attention:
            u64  = (self.up64(u32)+d64) * global_feat
        else:
            u64  = (self.up64(u32)+d64)
        u128 = self.up128(u64)+d128
        o_kp = torch.sigmoid(self.up512_kp(u128))
        o_ln = self.up512_ln(u128)
        
        return o_kp, o_ln, u128
