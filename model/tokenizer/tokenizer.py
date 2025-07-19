import torch
import torch.nn as nn
from tokenizer.feature_extractor.encoder import Encoder
from tokenizer.rvq.rvq import RVQ_VAE


class Tokenizer(nn.Module):
    def __init__(self, d_model = 256, latent_dim = 256, codebook_size = 256, num_codebooks = 3):
        super().__init__()
        self.encoder = Encoder()
        self.rvq = RVQ_VAE(latent_dim, codebook_size, num_codebooks)
        self.embdding_dim = nn.Embedding(codebook_size, d_model) # Feature embedding layer

    def forward(self, x):
        latents = self.encoder(x)
        code_index, codebook_losses, comitment_losses = self.rvq(latents)
        code_vector = self.embdding_dim(code_index)
        return code_vector
    
if __name__ == "__main__":
    x = torch.rand(4, 1, 160000)
    model = Tokenizer()
    y = model(x)
    print(y.shape)