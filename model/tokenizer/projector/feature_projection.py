import torch
import torch.nn as nn

class FeatureProjection(nn.Module):
    def __init__(self, in_channels, d_model = 256): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = d_model
        self.projection_conv = nn.Conv1d(in_channels, d_model, 1)
    
    def forward(self, x):
        x = self.projection_conv(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn(4, 512, 760)
    model = FeatureProjection(512)
    out = model(x) 
    print(out.shape)