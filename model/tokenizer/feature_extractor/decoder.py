import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer.components import TDConvBlock

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.td_conv_1 = TDConvBlock(256, 256, 9, 1)
        self.td_conv_2 = TDConvBlock(256, 128, 9, 1)
        self.td_conv_3 = TDConvBlock(128, 128, 25, 2, False)
        self.dropout = nn.Dropout1d(0.1)
        self.final_proj = nn.Conv1d(128, 1, kernel_size=52, stride=1, dilation=1, padding=0)
        
    def skip_connection(self, x, skip_connection):

        diff = skip_connection.shape[-1] - x.shape[-1]

        if diff > 0:
            x = F.pad(x, (0, diff))
        elif diff < 0:
            x = x[..., :skip_connection.shape[-1]]

        return x + skip_connection

    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]
        x = self.skip_connection(x, skip_connections[0])
        x = self.upsample(x)
        x = self.td_conv_1(x)
        x = self.skip_connection(x, skip_connections[1])
        x = self.upsample(x)
        x = self.td_conv_2(x)
        x = self.skip_connection(x, skip_connections[2])
        x = self.dropout(x)
        x = self.upsample(x)
        x = self.td_conv_3(x)
        x = self.skip_connection(x, skip_connections[3])
        x = self.upsample(x)
        x = self.final_proj(x)
        x = self.skip_connection(x, skip_connections[4])
        return x