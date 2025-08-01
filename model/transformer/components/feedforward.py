import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        hidden_size = config.embedding_dim * config.mlp_ratio
        self.intermediate_dense = nn.Linear(config.embedding_dim, hidden_size)
        self.activation = nn.GELU()
        self.intermediate_drop = nn.Dropout(config.hidden_dropout_p)

        self.output_dense = nn.Linear(hidden_size, config.embedding_dim)
        self.output_drop = nn.Dropout(config.hidden_dropout_p)

    def forward(self, x):
        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.intermediate_drop(x)
        x = self.output_dense(x)
        x = self.output_drop(x)

        return x