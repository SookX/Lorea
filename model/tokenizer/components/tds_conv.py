import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class TDSConvBlock(nn.Module):
    def __init__(self, i, c, k, s, dropout = True):
        super().__init__()
        self.pointwise_reverse = nn.ConvTranspose1d(i, i, kernel_size=1, stride=1)
        self.depthwise_reverse = nn.ConvTranspose1d(i, c, kernel_size=k, 
                                                    padding=k//2, stride=s, groups=c)
        self.batch_norm = nn.BatchNorm1d(c)
        self.activation = nn.LeakyReLU()
        self.is_dropout = dropout
        self.dropout_layer = nn.Dropout1d(0.15)
    
    def forward(self, x):
        x = self.pointwise_reverse(x)
        x = self.depthwise_reverse(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        if self.is_dropout:
            x = self.dropout_layer(x)
        return x