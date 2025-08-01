import torch
import torch.nn as nn
import torch.functional as F

class Attention(nn.Module):
    def __init__(self, conifg):
        super().__init__()

        self.config = conifg
        assert conifg.embedding_dim % conifg.num_attention_heads == 0, "Shape missmatch"

        self.head_dim = conifg.embedding_dim // conifg.num_attention_heads

        self.q_proj = nn.Linear(conifg.embedding_dim, conifg.embedding_dim)
        self.k_proj = nn.Linear(conifg.embedding_dim, conifg.embedding_dim)
        self.v_proj = nn.Linear(conifg.embedding_dim, conifg.embedding_dim)
        self.out_proj = nn.Linear(conifg.embedding_dim, conifg.embedding_dim)

    def forward(self, src, tgt = None, attention_mask = None, causal = False):
        batch, src_len, embed_dim = src.shape

        if tgt is None:
            q = self.q_proj(src).reshape(batch, src_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(src).reshape(batch, src_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(src).reshape(batch, src_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)

            if attention_mask is not None:
                attention_mask = attention_mask.bool() # [b, s]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, src_len, 1) # [b, 1, 1, s] -> dummy dimentions -> [b, 1, s, s]

            attention_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=self.config.attention_dropout_p if self.training else 0.0, is_causal=causal) # use flash attention
        else:
            tgt_len = tgt.shape[1]
            q = self.q_proj(tgt).reshape(batch, tgt_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(src).reshape(batch, src_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(src).reshape(batch, src_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)

            if attention_mask is not None:
                attention_mask = attention_mask.bool() # [b, s]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, tgt_len, 1) # [b, 1, 1, s] -> dummy dimentions -> [b, 1, t, s]

            attention_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=self.config.attention_dropout_p if self.training else 0.0, is_causal=False)
            
        attention_out = attention_out.transpose(1, 2).flatten(2)
        attention_out = self.out_proj(attention_out)