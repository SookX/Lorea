import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class DSConvBlock(nn.Module):
    def __init__(self, i, c, k, s, dropout = True):
        super().__init__()
        self.depthwise = nn.Conv1d(i, i, kernel_size=k, padding=k//2, stride=s, groups=i)
        self.pointwise = nn.Conv1d(i, c, kernel_size=1)
        self.activation = nn.LeakyReLU()
        self.batch_norm = nn.BatchNorm1d(c)
        self.is_dropout = dropout
        self.dropout_layer = nn.Dropout1d(0.15)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        if self.is_dropout:
            x = self.dropout_layer(x)
        return x