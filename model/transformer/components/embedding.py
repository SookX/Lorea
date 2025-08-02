import torch
import torch.nn as nn
from transformer.components.pos_enc import PositionalEncoding

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.src_embeddings = nn.Embedding(config.src_vocab_size, config.embedding_dim)
        self.tgt_embeddings = nn.Embedding(config.tgt_vocab_size, config.embedding_dim)

        self.src_positional_encodings = PositionalEncoding(config.max_src_len, config.embedding_dim, config.learn_pos_embed)
        self.tgt_positional_encodings = PositionalEncoding(config.max_tgt_len, config.embedding_dim, config.learn_pos_embed)

    def forward_src(self, input_ids):
        embeddings = self.src_embeddings(input_ids)
        embeddings = self.src_positional_encodings(embeddings)
        
        return embeddings
    
    def forward_tgt(self, target_ids):
        embeddings = self.tgt_embeddings(target_ids)
        embeddings = self.tgt_positional_encodings(embeddings)
        
        return embeddings