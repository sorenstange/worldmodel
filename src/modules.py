import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, max_len, condition_dim = 1, dropout = 0.1):
        super().__init__()
        self.embedding = Embedding(input_dim, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerLayer_AdaLN(
                d_model = d_model, 
                num_heads = num_heads,
                condition_dim = condition_dim,
                dropout = dropout) 
                for _ in range(num_layers)
            ])
                                    
        self.max_len = max_len

    def forward(self, x, cond):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if cond.dim() == 2:
            cond = cond.unsqueeze(-1)
        
        _, seq_len, _ = x.shape

        if seq_len > self.max_len:
            x = x[:, -self.max_len, :]
            seq_len = self.max_len

        mask = self.create_causal_mask(seq_len).to(x.device)

        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, cond, mask)

        return x

    def create_causal_mask(self, seq_len):
        return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, max_len, dropout = 0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) 
        self.embedding = Embedding(input_dim, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model = d_model, 
                num_heads = num_heads,
                dropout = dropout) 
                for _ in range(num_layers)
            ])
        
        self.max_len = max_len

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

class AdaLN(nn.Module):
    def __init__(self, hidden_dim, condition_dim=1): # F.eks. 1 for return + 3 for action-logits
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cond_to_scale_shift = nn.Linear(condition_dim, hidden_dim * 2)
        
        nn.init.zeros_(self.cond_to_scale_shift.weight)
        nn.init.zeros_(self.cond_to_scale_shift.bias)

    def forward(self, x, condition):
        normed_x = self.norm(x)
        
        if condition.dim() == 2:
            scale_shift = self.cond_to_scale_shift(condition).unsqueeze(1) # [B, 1, H*2]
        else:
            scale_shift = self.cond_to_scale_shift(condition) # [B, T, H*2]
            
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
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)

class TransformerLayer_AdaLN(nn.Module):
    def __init__(self, d_model, num_heads, condition_dim, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            dropout,
        )

        self.ffn = FeedForward(d_model, 2*d_model, dropout)

        self.norm1 = AdaLN(d_model, condition_dim)
        self.norm2 = AdaLN(d_model, condition_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, cond, mask=None):
        # Pre-LN på Attention grenen
        norm_x = self.norm1(x, cond)
        attn_out = self.self_attn(norm_x, mask)
        x = x + self.dropout1(attn_out)

        # Pre-LN på FeedForward grenen
        norm_x2 = self.norm2(x, cond)
        ff_out = self.ffn(norm_x2)
        x = x + self.dropout2(ff_out)

        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            dropout,
        )

        self.ffn = FeedForward(d_model, 2*d_model, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN på Attention grenen
        norm_x = self.norm1(x)
        attn_out = self.self_attn(norm_x, mask)
        x = x + self.dropout1(attn_out)

        # Pre-LN på FeedForward grenen
        norm_x2 = self.norm2(x)
        ff_out = self.ffn(norm_x2)
        x = x + self.dropout2(ff_out)

        return x
