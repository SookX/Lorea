import torch
import torch.nn as nn
from model.feature.fe import FeatureExtractor
from model.setcs.setcs import SETCS
from model.conformer.conformer import Conformer
import torch.nn.init as init

class Lorea(nn.Module):
    def __init__(self,
                 conv_feautre_channels: int = 288,
                 num_of_classes: int = 32):
        super().__init__()
        self.feature_extractor_ = FeatureExtractor(1024, 288)
        #self.setcs_list = nn.ModuleList([
        #    SETCS(in_channels=conv_feautre_channels, out_channels=conv_feautre_channels, kernel_size=33, stride=1, dropout=0.3, dillation=1),
        #    SETCS(in_channels=conv_feautre_channels, out_channels=conv_feautre_channels, kernel_size=41, stride=1, dropout=0.3, dillation=2),
        #    SETCS(in_channels=conv_feautre_channels, out_channels=conv_feautre_channels, kernel_size=41, stride=1, dropout=0.3, dillation=2),
        #    SETCS(in_channels=conv_feautre_channels, out_channels=conv_feautre_channels * 2, kernel_size=49, stride=1, dropout=0.3, dillation=4),
        #    SETCS(in_channels=conv_feautre_channels * 2, out_channels=conv_feautre_channels * 2, kernel_size=57, stride=1, dropout=0.3, dillation=8),
        #])
        self.conformer = Conformer(conv_feautre_channels, 12, conv_feautre_channels * 2, 5, 21, 0.2, True)
        self.proj_layer = nn.Linear(conv_feautre_channels, num_of_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, seq_lens):
        x, s1, s2, output_seq_lens, mask = self.feature_extractor_(x, seq_lens) # [b, c, f]
        #print(x.shape)
        #x = self.setcs_list[0](x, mask)
        #for module in self.setcs_list[1:]:
        #    out = module(x, mask)
        #    if x.shape[1] != out.shape[1]:
        #        x = nn.functional.pad(x, (0,0,0,out.shape[1]-x.shape[1]))
        #    x = x + out
#
        x = x.permute(0, 2, 1) # [b, f, c]
        output_frames, output_lenghts = self.conformer(x, output_seq_lens)
        proj = self.proj_layer(output_frames)
        return proj, output_lenghts, s1, s2
