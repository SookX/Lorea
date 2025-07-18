import torch
import torch.nn as nn
from fextractor import FeatureExtractor, Decoder

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

class RVQ_VAE(nn.Module):
    def __init__(self, latent_dim, codebook_size = 512, num_codebooks = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.enc = FeatureExtractor()
        self.dec = Decoder()
        self.rvq = nn.ModuleList(
            [
                VectorQuantizer(codebook_size, latent_dim) for _ in range(num_codebooks)
            ]
        )
        self.gpool = nn.AdaptiveAvgPool1d(1)
        self.ff = nn.Linear(256, 35)

    def quantize(self, z, return_code_idx_only = False):
        codebook_losses = 0
        comitment_losses = 0

        quantized_codes = []
        code_index = []

        final_quantized = torch.zeros_like(z)

        for quantizer in self.rvq:
            codes, code_idx = quantizer(z)
            code_index.append(code_idx)

            codebook_loss = torch.mean((codes - z.detach())**2)
            comitment_loss = torch.mean((codes.detach() - z) ** 2)

            codebook_losses += codebook_loss
            comitment_losses += comitment_loss
 
            codes = z + (codes - z).detach()

            final_quantized = final_quantized + codes
            quantized_codes.append(codes)

            z = z - codes.detach()
        
        if return_code_idx_only:
            return torch.stack(code_index, dim = 1)
        
        else:
            return final_quantized, codebook_losses, comitment_losses

    def forward_enc(self, x):
        return self.enc(x)
    
    def forward_dec(self, x, skip_connections):
        return self.dec(x, skip_connections)
    
    def forward(self, x):
        latents, skip_connections = self.forward_enc(x)          # [B, D, T]
        latents = latents.permute(0, 2, 1)     # [B, T, D]
        B, T, D = latents.shape

        latents_flat = latents.reshape(B * T, D)
        final_quantized, codebook_losses, comitment_losses = self.quantize(latents_flat)

        final_quantized = final_quantized.reshape(B, T, D)
        final_quantized = final_quantized.permute(0, 2, 1)
        
        decoded = self.forward_dec(final_quantized, skip_connections)

        output = self.gpool(final_quantized).squeeze(-1)
        output = self.ff(output)

        return output, decoded, codebook_losses, comitment_losses


    
if __name__ == "__main__":
    x = torch.rand(4, 1, 100000)
    model = RVQ_VAE(256)
    output, codebook_losses, comitment_losses = model(x)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {pytorch_total_params}")
    #print(output.shape)