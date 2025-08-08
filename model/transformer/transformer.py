import torch
import torch.nn as nn
from model.transformer.components.embedding import Embeddings
from model.transformer.encoder.encoder import TransformerEncoder
from model.transformer.decoder.decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.encoder = nn.ModuleList(
            [TransformerEncoder(config) for _ in range(config.encoder_depth)]
        )
        self.decoder = nn.ModuleList(
            [TransformerDecoder(config) for _ in range(config.decoder_depth)]
        )
        self.head = nn.Linear(config.embedding_dim, config.tgt_vocab_size)
    
    def forward(self, src_ids, tgt_ids, tgt_mask = None):
        src_embeddings = self.embeddings.forward_src(src_ids).permute(0, 2, 1)
        tgt_embeddings = self.embeddings.forward_tgt(tgt_ids)

        for layer in self.encoder:
            src_embeddings = layer(src_embeddings)
        
        for layer in self.decoder:
            tgt_embeddings = layer(src_embeddings, tgt_embeddings, None, tgt_mask)

        pred = self.head(tgt_embeddings)
        return pred




if __name__ == '__main__':
    src_ids = torch.rand(4, 4000)

    