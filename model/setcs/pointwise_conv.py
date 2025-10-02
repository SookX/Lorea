import torch
import torch.nn as nn

class GroupShuffle(nn.Module):

    def __init__(self, groups, channels):
        super(GroupShuffle, self).__init__()

        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(self, x):
        sh = x.shape
        x = x.view(-1, self.groups, self.channels_per_group, sh[-1])


        x = torch.transpose(x, 1, 2).contiguous()


        x = x.view(-1, self.groups * self.channels_per_group, sh[-1])
        x = x.view(sh)

        return x

class PointwiseConvolution(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels
                 ):
        super().__init__()
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, groups=in_channels)
        #elf.gropu_shuffle = GroupShuffle(in_channels, in_channels)



    def forward(self, x):
        x = self.pointwise(x)
        return x