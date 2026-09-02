import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, max_len,
                 condition_dim=1, dropout=0.1, ff_mult=4):
        super().__init__()
        self.embedding = Embedding(input_dim, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerLayer_AdaLN(
                d_model=d_model,
                num_heads=num_heads,
                condition_dim=condition_dim,
                dropout=dropout,
                ff_mult=ff_mult)
            for _ in range(num_layers)
        ])

        self.max_len = max_len
        _depth_scaled_init(self.layers)

    def forward(self, x, cond):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if cond.dim() == 2:
            cond = cond.unsqueeze(-1)

        # Truncate x and cond together; they must stay aligned on the time axis.
        if x.size(1) > self.max_len:
            x = x[:, -self.max_len:, :]
            if cond.dim() == 3 and cond.size(1) > self.max_len:
                cond = cond[:, -self.max_len:, :]

        x = self.embedding(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, cond, is_causal=True)

        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, max_len,
                 dropout=0.1, ff_mult=4):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.embedding = Embedding(input_dim, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ff_mult=ff_mult)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

        self.max_len = max_len
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        _depth_scaled_init(self.layers)

    def forward(self, x):
        if x.dim() < 3:
            x = x.unsqueeze(0)
        batch, _, _ = x.size()
        cls_tokens = self.cls_token.expand(batch, -1, -1)

        x = self.embedding(x)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pe(x)

        for layer in self.layers:
            x = layer(x)

        # Final norm before read-out: a pre-LN stack leaves the residual stream
        # unnormalised, which matters here because the CLS row is the latent.
        x = self.norm_out(x)

        return x[:, 0, :]


class SIGReg(nn.Module):
    """Sketched Epps-Pulley Gaussianity statistic (LeJEPA).

    Penalises how far the batch distribution of random 1-D projections of the
    latents is from a standard Gaussian. This is what keeps the encoder from
    collapsing to a constant, since the MSE term on its own has that trivial
    solution. Restored from 1c57f96 -- see CLAUDE.md.
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, z):
        """z: (B, T, D). The Gaussianity test runs across the batch axis, per
        (timestep, projection) pair."""
        proj = z.permute(1, 0, 2)  # (T, B, D)

        A = torch.randn(proj.size(-1), self.num_proj,
                        device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))

        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class Embedding(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)

    def forward(self, x):
        return self.projection(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)

    def forward(self, x, offset=0):
        seq_len = x.size(1)
        positions = torch.arange(
            offset,
            offset + seq_len,
            device=x.device
        )
        pos = self.embedding(positions)
        return x + pos.unsqueeze(0)


class AdaLN(nn.Module):
    def __init__(self, hidden_dim, condition_dim=1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cond_to_scale_shift = nn.Linear(condition_dim, hidden_dim * 2)

        nn.init.zeros_(self.cond_to_scale_shift.weight)
        nn.init.zeros_(self.cond_to_scale_shift.bias)

    def forward(self, x, condition):
        normed_x = self.norm(x)

        if condition.dim() == 2:
            scale_shift = self.cond_to_scale_shift(condition).unsqueeze(1)  # [B, 1, H*2]
        else:
            scale_shift = self.cond_to_scale_shift(condition)  # [B, T, H*2]

        gamma, beta = scale_shift.chunk(2, dim=-1)
        return normed_x * (1.0 + gamma) + beta


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout_p = dropout

    def forward(self, x, is_causal=False):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # SDPA picks a fused/flash kernel where available. A single-token
        # sequence has nothing to mask, and is_causal=True would be a no-op.
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal and seq_len > 1,
        )

        out = out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_ff, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerLayer_AdaLN(nn.Module):
    def __init__(self, d_model, num_heads, condition_dim, dropout=0.1, ff_mult=4):
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, ff_mult * d_model, dropout)

        self.norm1 = AdaLN(d_model, condition_dim)
        self.norm2 = AdaLN(d_model, condition_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, cond, is_causal=False):
        # Pre-LN on the attention branch
        norm_x = self.norm1(x, cond)
        attn_out = self.self_attn(norm_x, is_causal=is_causal)
        x = x + self.dropout1(attn_out)

        # Pre-LN on the feed-forward branch
        norm_x2 = self.norm2(x, cond)
        ff_out = self.ffn(norm_x2)
        x = x + self.dropout2(ff_out)

        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, ff_mult=4):
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, ff_mult * d_model, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, is_causal=False):
        # Pre-LN on the attention branch
        norm_x = self.norm1(x)
        attn_out = self.self_attn(norm_x, is_causal=is_causal)
        x = x + self.dropout1(attn_out)

        # Pre-LN on the feed-forward branch
        norm_x2 = self.norm2(x)
        ff_out = self.ffn(norm_x2)
        x = x + self.dropout2(ff_out)

        return x


def _depth_scaled_init(layers):
    """Scale each residual branch's output projection by 1/sqrt(2*depth).

    Without this the residual stream variance grows linearly with depth, which
    is exactly the regime this project runs in (see the depth/width note in
    CLAUDE.md).
    """
    depth = len(layers)
    if depth == 0:
        return
    scale = 1.0 / math.sqrt(2.0 * depth)
    for layer in layers:
        nn.init.normal_(layer.self_attn.out_proj.weight, mean=0.0, std=0.02 * scale)
        nn.init.zeros_(layer.self_attn.out_proj.bias)
        nn.init.normal_(layer.ffn.net[-1].weight, mean=0.0, std=0.02 * scale)
        nn.init.zeros_(layer.ffn.net[-1].bias)
