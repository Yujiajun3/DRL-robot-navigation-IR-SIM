import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiheadAttention(embed_size, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attention_output, attention_weights = self.attention(query, key, value)

        x = self.dropout(self.norm1(attention_output + query)) # add residual, normalize and do dropout
        forward = self.feed_forward(x) # add another mlp, regain the size
        out = self.dropout(self.norm2(forward + x))  # another residual, normalize and do dropout
        return out

# Encoder
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_size,
        heads,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.transformer = TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )

        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence):
        query = sequence[:,-1,:].unsqueeze(1)
        _, hist_len, _ = sequence.shape
        # print(f"Query shape: {query.shape}")
        key_value = sequence[:, :-1, :]
        # print(f"Key-Value shape: {key_value.shape}")
        N, seq_length,_ = key_value.shape # batch size, max length of sequence
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(query.device)
        # print(f"Positions shape: {positions.shape}")# positional encoder
        key_value = key_value + self.position_embedding(positions) # apply positional encoder for each embedding in respective sequence, have dropout

        out = self.transformer(query, key_value, key_value)
        return out.squeeze(1)