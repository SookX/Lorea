import torch
import torch.nn as nn
from model.transformer.components.attention import Attention
from model.transformer.components.feedforward import FeedForward

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dec_attention = Attention(config)
        self.dec_attention_drop = nn.Dropout(config.hidden_dropout_p)
        self.dec_attention_layernorm = nn.LayerNorm(config.embedding_dim)

        self.cross_attention = Attention(config)
        self.cross_attention_drop = nn.Dropout(config.hidden_dropout_p)
        self.cross_attention_layernorm = nn.LayerNorm(config.embedding_dim)

        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, src, tgt, src_mask=None, tgt_mask = None):
        tgt = tgt + self.dec_attention_drop(self.dec_attention(tgt, attention_mask = tgt_mask, causal=True))
        tgt = self.dec_attention_layernorm(tgt)

        tgt = tgt + self.cross_attention_drop(self.cross_attention(src, tgt, attention_mask = src_mask))
        tgt = self.cross_attention_layernorm(tgt)

        tgt = tgt + self.feed_forward(tgt)
        tgt = self.final_layer_norm(tgt)

        return tgt