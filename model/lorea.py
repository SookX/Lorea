import torch
import torch.nn as nn
from model.feature.fe import FeatureExtractor
from model.setcs.setcs import SETCS
from torchaudio.models import Conformer

class Lorea(nn.Module):
    def __init__(self,
                 conv_feautre_channels: int = 144,
                 num_of_classes: int = 32):
        super().__init__()
        self.feature_extractor_ = FeatureExtractor(conv_feautre_channels = conv_feautre_channels)
        self.setcs_list = nn.ModuleList([
            SETCS(in_channels=conv_feautre_channels, out_channels=conv_feautre_channels, kernel_size=33, stride=1, dropout=0.3, dillation=1),
            SETCS(in_channels=conv_feautre_channels, out_channels=conv_feautre_channels, kernel_size=41, stride=1, dropout=0.3, dillation=2),
            SETCS(in_channels=conv_feautre_channels, out_channels=conv_feautre_channels, kernel_size=41, stride=1, dropout=0.3, dillation=2),
            SETCS(in_channels=conv_feautre_channels, out_channels=conv_feautre_channels * 2, kernel_size=49, stride=1, dropout=0.3, dillation=4),
            SETCS(in_channels=conv_feautre_channels * 2, out_channels=conv_feautre_channels * 2, kernel_size=57, stride=1, dropout=0.3, dillation=8),
        ])
        self.conformer = Conformer(conv_feautre_channels * 2, 4, 576, 5, 31, 0.2)
        self.proj_layer = nn.Linear(conv_feautre_channels * 2, num_of_classes)

    def forward(self, x, seq_lens):
        x, output_seq_lens, mask = self.feature_extractor_(x, seq_lens) # [b, c, f]
        #print(x.shape)
        x = self.setcs_list[0](x, mask)
        for module in self.setcs_list[1:]:
            out = module(x, mask)
            if x.shape[1] != out.shape[1]:
                x = nn.functional.pad(x, (0,0,0,out.shape[1]-x.shape[1]))
            x = x + out

        x = x.permute(0, 2, 1) # [b, f, c]
        output_frames, output_lenghts = self.conformer(x, output_seq_lens)
        proj = self.proj_layer(output_frames)
        return proj, output_lenghts
