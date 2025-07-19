import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dropout = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        _, wave_length, d_model = x.shape
        pe = torch.zeros(wave_length, d_model)
        position = torch.arange(0, wave_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) 
        return self.dropout(x)
    
if __name__ == "__main__":
    x = torch.rand(4, 3100, 256)
    pos = PositionalEncoding()
    x = pos(x)
    print(x.shape)

