import torch
import torch.nn as nn

class DepthwiseConvolution(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size,
                 stride=1,
                 dilation=1):
        super().__init__()

        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=((kernel_size-1)*dilation)//2,
            groups=in_channels,
            dilation=dilation
        )

    def forward(self, x):
        return self.depthwise(x)
