import torch
import torch.nn as nn
from tokenizer.components.ds_conv import DSConvBlock
from tokenizer.components.sinc_conv import SincBlock

class Encoder(nn.Module):
    def __init__(self, return_skip_connections=False):
        super().__init__()
        self.return_skip_connections = return_skip_connections

        self.sinc_block = SincBlock(128, 52)

        self.d_conv_1 = DSConvBlock(128, 128, 25, 2, False)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout1d(0.1)
        self.d_conv_2 = DSConvBlock(128, 256, 9, 1, False)
        self.d_conv_3 = DSConvBlock(256, 256, 9, 1, False)
        
    def forward(self, x):
        if self.return_skip_connections:
            skip_connections = []
            skip_connections.append(x)

        x = self.sinc_block(x)

        if self.return_skip_connections:
            skip_connections.append(x)

        x = self.d_conv_1(x)
        x = self.pool(x)
        x = self.dropout(x)

        if self.return_skip_connections:
            skip_connections.append(x)

        x = self.d_conv_2(x)
        x = self.pool(x)

        if self.return_skip_connections:
            skip_connections.append(x)

        x = self.d_conv_3(x)
        x = self.pool(x)

        if self.return_skip_connections:
            skip_connections.append(x)

        if self.return_skip_connections:
            return x, skip_connections
        else:
            return x

if __name__ == "__main__":
    waveform = torch.rand(4, 1, 640000)
    model = Encoder()
    y = model(waveform)
    print(y.shape)
