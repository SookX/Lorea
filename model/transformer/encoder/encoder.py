import torch
import torch.nn as nn
from transformer.components.attention import Attention
from transformer.components.feedforward import FeedForward


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder_attention = Attention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_p)

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder_attention = Attention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_p)
        self.layer_norm = nn.LayerNorm(config.embedding_dim)

        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, x, attention_mask = None):
       
       x = self.encoder_attention(x, attention_mask = attention_mask)
       x = x + self.dropout(x)
       x = self.layer_norm(x)

       x = x + self.feed_forward(x)
       x = self.final_layer_norm(x)
       return x