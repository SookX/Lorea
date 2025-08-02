import torch
import torch.nn as nn
from tokenizer.tokenizer import Tokenizer
from config import TransformerConfig
from transformer.transformer import Transformer

class Lorea(nn.Module):
    def __init__(self, latent_dim = 256, codebook_size = 256, num_codebooks = 3):
        super().__init__()
        self.tok = Tokenizer( latent_dim, codebook_size, num_codebooks)
        self.transformer = Transformer(TransformerConfig())
        
       
    
    def forward(self, waveform, tgt_ids, tgt_mask = None):
        latents = self.tok(waveform)
        out = self.transformer(latents, tgt_ids, tgt_mask)
        return out

if __name__ == '__main__':
    x = torch.rand(4, 1, 160000)
    tgt_ids = torch.randint(0, 1000, size=(4, 500))
    model = Lorea()
    y = model(x, tgt_ids)
    print(y.shape)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {pytorch_total_params}")
