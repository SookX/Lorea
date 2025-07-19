import torch
import torch.nn as nn
from tokenizer.feature_extractor.encoder import Encoder
from tokenizer.feature_extractor.decoder import Decoder
from tokenizer.components.vq import VectorQuantizer

class RVQ_VAE(nn.Module):
    def __init__(self, latent_dim, codebook_size = 512, num_codebooks = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.enc = Encoder()
        # self.dec = Decoder()
        self.rvq = nn.ModuleList(
            [
                VectorQuantizer(codebook_size, latent_dim) for _ in range(num_codebooks)
            ]
        )
        # self.gpool = nn.AdaptiveAvgPool1d(1)
        # self.ff = nn.Linear(256, 35)

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
            return final_quantized,code_index, codebook_losses, comitment_losses

    def forward_enc(self, x):
        return self.enc(x)
    
    #def forward_dec(self, x, skip_connections):
    #    return self.dec(x, skip_connections)
    
    def forward(self, x):
        latents = self.forward_enc(x)          # [B, D, T]
        latents = latents.permute(0, 2, 1)     # [B, T, D]
        B, T, D = latents.shape

        latents_flat = latents.reshape(B * T, D)
        final_quantized, code_index, codebook_losses, comitment_losses = self.quantize(latents_flat)
        #final_quantized = final_quantized.reshape(B, T, D)
        #final_quantized = final_quantized.permute(0, 2, 1)
        
        code_index = code_index[-1].reshape(B, -1)
        #decoded = self.forward_dec(final_quantized, skip_connections)

        #output = self.gpool(final_quantized).squeeze(-1)
        #output = self.ff(output)

        return code_index, codebook_losses, comitment_losses


    
if __name__ == "__main__":
    x = torch.rand(4, 1, 100000)
    model = RVQ_VAE(256)
    code_index, codebook_losses, comitment_losses = model(x)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {pytorch_total_params}")
    print(code_index.shape)