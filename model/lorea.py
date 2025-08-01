import torch
import torch.nn as nn
from tokenizer.tokenizer import Tokenizer

class Lorea(nn.Module):
    def __init__(self, d_model = 256, latent_dim = 256, codebook_size = 256, num_codebooks = 3):
        super().__init__()
        self.tok = Tokenizer(d_model, latent_dim, codebook_size, num_codebooks)
       
    
    def forward(self, x):
        x = self.tok(x)
        return x

if __name__ == '__main__':
    x = torch.rand(4, 1, 160000)
    model = Lorea()
    y = model(x)
    print(y.shape)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {pytorch_total_params}")
