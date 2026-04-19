import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnedPositionalEncoding(nn.Module):
    """
    Fix #3: Replaced standard sinusoidal positional encoding with a learned embedding.

    Rationale: The original sinusoidal PE adds dimension-correlated noise on top of raw
    tabular stats. IPL teams have no natural linguistic sequence; the only ordering is
    league standings (already captured by the `position` feature). A learned, small
    positional embedding lets the model discover whether stand-table rank matters for
    each dimension, rather than imposing a fixed trigonometric pattern that has no
    cricket-domain meaning.
    """
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        return x + self.embedding(positions)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear projections
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        # Concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(output)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),          # GELU is standard in modern transformers (vs ReLU)
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Sublayer 1: Multi-Head Attention with residual
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Sublayer 2: Feed Forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class IPLPredictorTransformer(nn.Module):
    """
    Decoder-only Transformer for IPL winner prediction.

    Input:  (batch, n_teams, input_dim) — one row per team, input_dim raw features.
    Output: (batch, n_teams) — softmax probabilities summing to 1.

    Features (input_dim=7, excludes won_title to prevent data leakage):
        wins, losses, nrr, points, position, historical_titles, qualified_playoffs
    """
    def __init__(self, input_dim=7, d_model=128, n_heads=8, n_layers=4, d_ff=512, max_teams=10):
        super().__init__()
        # Project raw statistics to d_model
        self.embedding = nn.Linear(input_dim, d_model)

        # Fix #3: Learned positional encoding (replaces sinusoidal PE)
        self.pos_encoding = LearnedPositionalEncoding(max_len=max_teams, d_model=d_model)

        # Decoder-only transformer stack
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        # Output head: logit per team → softmax probability
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x, use_causal_mask=False):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            use_causal_mask: Apply look-ahead mask (useful during autoregressive training).
                             For inference over a fixed set of n teams, set to False.
        """
        batch_size, seq_len, _ = x.size()

        mask = None
        if use_causal_mask:
            mask = torch.tril(torch.ones((seq_len, seq_len))).view(
                1, 1, seq_len, seq_len
            ).to(x.device)

        x = self.embedding(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        # (batch, seq_len, 1) → (batch, seq_len)
        logits = self.output_head(x).squeeze(-1)
        return F.softmax(logits, dim=-1)


def build_model(input_dim=7):
    return IPLPredictorTransformer(input_dim=input_dim)


if __name__ == "__main__":
    model = build_model(7)
    # 1 batch, 10 teams, 7 features each (no won_title)
    sample_input = torch.randn(1, 10, 7)
    probs = model(sample_input)
    print(f"Prediction probabilities for 10 teams:\n{probs}")
    print(f"Sum (should be ~1.0): {probs.sum().item():.6f}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
