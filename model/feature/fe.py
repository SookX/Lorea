import torch
import torch.nn as nn
from model.feature.masked_conv import MaskedConv1D
from model.feature.cbam import CBAM1D

class FeatureExtractor(nn.Module): # Simmilar to Whisper module
    def __init__(self, conv_feautre_channels):
        super().__init__()

        self.conv1 = MaskedConv1D(80, conv_feautre_channels, kernel_size=5, stride=2, padding=1)
        self.conv2 = MaskedConv1D(conv_feautre_channels, conv_feautre_channels, kernel_size=5, stride=1, padding=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.cbam = CBAM1D(conv_feautre_channels)

    def forward(self, x, seq_lens):

        x, output_seq_lens = self.conv1(x, seq_lens)
        x = self.activation(x)
        x = self.dropout(x)
        x, output_seq_lens, mask = self.conv2(x, output_seq_lens, return_mask = True)
        x = self.activation(x)
        
        x = self.cbam(x)

        x = self.dropout(x)
        x = x * mask

        return x, output_seq_lens, mask

    
        
    
if __name__ == "__main__":
    x =torch.rand(3, 80, 1400)
    fe = FeatureExtractor()
    print(fe(x).shape)