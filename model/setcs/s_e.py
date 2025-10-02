import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    def __init__(self,
                 in_channels,
                 reduction_ratio = 4,
                 activation = nn.ReLU,
                 scale_activation = nn.Sigmoid):
        
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.squeeze_channels = max(1, in_channels // reduction_ratio)
        self.fc1 = nn.Conv1d(in_channels, self.squeeze_channels, 1)
        self.fc2 = nn.Conv1d(self.squeeze_channels, in_channels, 1)

        self.activation = activation()

        self.scale_activation = scale_activation()
        
    def _scale(self, input):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)
    
    def forward(self, x):
        scale = self._scale(x)
        return scale * x