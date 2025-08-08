import torch
import torch.nn as nn
from model.tokenizer.feature_extractor.encoder import Encoder
from model.tokenizer.rvq.rvq import RVQ_VAE
from model.tokenizer.projector.feature_projection import FeatureProjection


class Tokenizer(nn.Module):
    def __init__(self, latent_dim = 256, codebook_size = 256, num_codebooks = 3):
        super().__init__()
        self.encoder = Encoder()
        self.proj = FeatureProjection(256)
#        self.rvq = RVQ_VAE(latent_dim, codebook_size, num_codebooks)

    def forward(self, x):
        latents = self.encoder(x)
        latents = self.proj(latents)
        #code_index, codebook_losses, comitment_losses = self.rvq(latents)
        return latents
    
if __name__ == "__main__":
    x = torch.rand(4, 1, 160000)
    model = Tokenizer()
    y = model(x)
    print(y.shape)