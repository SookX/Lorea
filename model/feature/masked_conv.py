import torch
import torch.nn as nn

class MaskedConv1D(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding = 0,
                 bias = True,
                 **kwargs):
        super().__init__(in_channels=in_channels, 
                                           out_channels=out_channels, 
                                           kernel_size=kernel_size, 
                                           stride=stride, 
                                           padding=padding, 
                                           bias=bias, 
                                           **kwargs)
    def _compute_output_seq_len(self, seq_lens):
        """
        To perform masking AFTER the encoding 1D Convolutions, we need to 
        compute what the shape of the output tensor is after each successive convolutions
        is applied.
    
        Convolution formula can be found in PyTorch Docs: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        """
        
        return ((seq_lens + 2*self.padding[0] - (self.kernel_size[0]-1) - 1) // self.stride[0] + 1).long()

    def forward(self, x, seq_lens, return_mask = False):
        output_seq_lens = self._compute_output_seq_len(seq_lens)
        
        batch_size, channels, features = x.shape
        conv_out = super().forward(x)
        
        max_len = output_seq_lens.max()
        range_tensor = torch.arange(max_len, device=x.device).unsqueeze(0) 
        mask = (range_tensor < output_seq_lens.unsqueeze(1)).unsqueeze(1) 
        conv_out = conv_out * mask

        if return_mask:
            return conv_out, output_seq_lens, mask
        else:
            return conv_out, output_seq_lens