import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, Q, K, V):
        d_k = Q.size(-1)

        transposed_K = K.transpose(-2, -1)
        scores = torch.matmul(Q, transposed_K) / d_k
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, V)

        return output, scores