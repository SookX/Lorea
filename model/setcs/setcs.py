import torch
import torch.nn as nn
from model.setcs.depthwise_conv import DepthwiseConvolution
from model.setcs.pointwise_conv import PointwiseConvolution
from model.setcs.s_e import SqueezeExcitation


class SETCS(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 dillation,
                 dropout=0.2
                 ):
        super().__init__()
        self.depthwise_conv = DepthwiseConvolution(in_channels, kernel_size, stride, dillation)
        self.pointwise_conv = PointwiseConvolution(in_channels, out_channels)
        self.activation = nn.LeakyReLU()
        self.normalization = nn.BatchNorm1d(out_channels)
        self.se_layer = SqueezeExcitation(out_channels)
        self.dropout = nn.Dropout(p=dropout)
        if(in_channels == out_channels):
            self.residual = True
        else:
            self.residual = False

    def forward(self, input, mask: int = None):
        x = self.depthwise_conv(input)
        x = self.pointwise_conv(x)
        x = self.normalization(x)
        x = self.se_layer(x)
        if mask is not None:
            x = x * mask

        x = self.activation(x)
        x = self.dropout(x)
        if self.residual:
            x = x + input
        return x