import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Sin/Cosine (non-learnable) encodings proposed in Attention is All You Need.
    Dynamically computes positional encodings during the forward pass.
    
    Args:
        embed_dim: Embedding dimension of each token
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            x + positional encodings of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.size()

        position = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, dtype=torch.float32, device=x.device) *
                             (-math.log(10000.0) / self.embed_dim))  

        pe = torch.zeros(seq_len, self.embed_dim, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return x + pe.unsqueeze(0)  
