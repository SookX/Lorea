from dataclasses import dataclass

@dataclass
class TransformerConfig:

    embedding_dim: int = 256
    num_attention_heads: int = 8
    attention_dropout_p: float = 0.0
    hidden_dropout_p: float = 0.0
    mlp_ratio: int = 4
    encoder_depth: int = 3
    decoder_depth: int = 3

    src_vocab_size: int = 256
    tgt_vocab_size: int = 5000

    max_src_len: int = 15000
    max_tgt_len: int = 512
    learn_pos_embed: bool = False