import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=2, dim_feedforward=512, dropout=0.2, num_classes=6, task_type='sequence'):
        """
        Args:
            vocab_size (int): Size of the tokenizer vocabulary.
            d_model (int): Embedding dimension.
            nhead (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            dim_feedforward (int): Hidden layer size in feedforward network.
            dropout (float): Dropout probability.
            num_classes (int): Number of output classes.
            task_type (str): 'token' for per-token classification, 'sequence' for overall label.
        """
        super(TransformerModel, self).__init__()
        self.task_type = task_type

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # ✅ Use pre-norm architecture (LayerNorm before attention/FFN)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)  # ✅ Final LayerNorm
        self.dropout = nn.Dropout(dropout)

        if task_type == 'token':
            self.classifier = nn.Linear(d_model, num_classes)
        elif task_type == 'sequence':
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Linear(d_model, num_classes)
        else:
            raise ValueError("task_type must be 'token' or 'sequence'")

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length)
            attention_mask: Optional mask (batch_size, seq_length)
        """
        x = self.embedding(x)                            # (B, S, D)
        x = self.positional_encoding(x)                 # (B, S, D)
        x = self.transformer_encoder(x)                 # (B, S, D)
        x = self.norm(x)                                # ✅ Final LayerNorm
        x = self.dropout(x)                             # ✅ Final dropout

        if self.task_type == 'token':
            return self.classifier(x)                   # (B, S, num_classes)
        else:  # sequence classification
            pooled = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, D)
            return self.classifier(pooled)              # (B, num_classes)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
