import torch
import torch.nn as nn

class TransformerCircuitOptimizer(nn.Module):
    def __init__(self, vocab_size, emb_size=128, num_heads=4, num_layers=3, dim_feedforward=256):
        super(TransformerCircuitOptimizer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=num_heads, dim_feedforward=dim_feedforward
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out = self.transformer(emb)
        return self.decoder(out)
