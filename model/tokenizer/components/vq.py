import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size=1024, latent_dim=2):
        super().__init__()
        
        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.embedding.weight.data.uniform_(-1/codebook_size, 1/codebook_size)

        self.latent_dim = latent_dim
        self.codebook_size = codebook_size

    def forward(self, x):

        batch_size = x.shape[0]
        
        ### Distance btwn every Latent and Code: (L-C)**2 = (L**2 - 2LC + C**2 ) ###

        ### L2: [B, L] -> [B, 1]
        L2 = torch.sum(x**2, dim=1, keepdim=True)

        ### C2: [C, L] -> [C]
        C2 = torch.sum(self.embedding.weight**2, dim=1).unsqueeze(0)

        ### CL: [B,L]@[L,C] -> [B, C]
        CL = x@self.embedding.weight.t()

        ### [B, 1] - 2 * [B, C] + [C] -> [B, C]
        distances = L2 - 2*CL + C2
        
        ### Grab Closest Indexes, create matrix of corresponding vectors ###
        ### Closest: [B, 1]
        closest = torch.argmin(distances, dim=-1)

        ### Create Empty Quantized Latents Embedding ###
        # latents_idx: [B, C]
        quantized_latents_idx = torch.zeros(batch_size, self.codebook_size, device=x.device)

        ### Place a 1 at the Indexes for each sample for the codebook we want ###
        batch_idx = torch.arange(batch_size)
        quantized_latents_idx[batch_idx,closest] = 1

        ### Matrix Multiplication to Grab Indexed Latents from Embeddings ###

        # quantized_latents: [B, C] @ [C, L] -> [B, L]
        quantized_latents = quantized_latents_idx @ self.embedding.weight

        return quantized_latents, closest 