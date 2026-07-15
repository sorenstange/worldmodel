import torch
import torch.nn as nn

class Predictor(nn.module):
    def __init__(self, d_model, num_layers, num_heads, max_len, dropout = 0.1):
        super().__init__()
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model = d_model, 
                                    num_heads = num_heads,
                                    dim_ff = d_model * 2, 
                                    dropout = dropout) 
                                    for _ in range(num_layers)
                                    ])

    def forward(self, x):
        if x.dim() < 3:
            x = x.unsqueeze(0)
        _, seq_len, _ = x.shape
        mask = self.create_causal_mask(seq_len)

        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)

        return x

    def create_causal_mask(self, seq_len):
        return torch.tril(torch.ones(seq_len, seq_len))

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, max_len, dropout = 0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) 

        self.embedding = Embedding(input_dim, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model = d_model, 
                                    num_heads = num_heads,
                                    dim_ff = d_model * 2, 
                                    dropout = dropout) 
                                    for _ in range(num_layers)
                                    ])

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
        
        x = x[:, 0, :]

        return x

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

    def forward(self, x, offset=0):
        seq_len = x.size(1)
        positions = torch.arange(
            offset,
            offset + seq_len,
            device=x.device
        )
        pos = self.embedding(positions)
        return x + pos.unsqueeze(0)

class SIGReg(torch.nn.Module):
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

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time

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

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_ff, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            dropout,
        )

        self.ffn = FeedForward(d_model, dim_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dim_ff,
        dropout=0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    num_heads,
                    dim_ff,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
