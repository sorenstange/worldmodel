import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, max_len, num_bins, dropout = 0.1):
        super().__init__()
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerPredictorLayer(d_model = d_model, 
                                    num_heads = num_heads,
                                    dropout = dropout) 
                                    for _ in range(num_layers)
                                    ])
        self.return_head = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.LayerNorm(2*d_model),       
            nn.SiLU(),
            nn.Dropout(dropout),         
            nn.Linear(2*d_model, num_bins)
        )

    def forward(self, x, ret):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        _, seq_len, _ = x.shape
        mask = self.create_causal_mask(seq_len).to(x.device)

        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, ret, mask)

        return x, self.return_head(x)

    def create_causal_mask(self, seq_len):
        return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, max_len, dropout = 0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) 

        self.embedding = Embedding(input_dim, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model = d_model, 
                                    num_heads = num_heads,
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

class AdaLN(nn.Module):
    def __init__(self, hidden_dim, action_dim=1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.action_to_scale_shift = nn.Linear(action_dim, hidden_dim * 2)
        
        nn.init.zeros_(self.action_to_scale_shift.weight)
        nn.init.zeros_(self.action_to_scale_shift.bias)

    def forward(self, x, action):
        normed_x = self.norm(x)
        if action.dim() == 2:
            scale_shift = self.action_to_scale_shift(action).unsqueeze(1) # [B, 1, H*2]
        else:
            scale_shift = self.action_to_scale_shift(action) # [B, T, H*2]
            
        gamma, beta = scale_shift.chunk(2, dim=-1)
        return normed_x * (1.0 + gamma) + beta

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
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)

class TransformerPredictorLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            dropout,
        )

        self.ffn = FeedForward(d_model, 2*d_model, dropout)

        self.norm1 = AdaLN(d_model)
        self.norm2 = AdaLN(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, ret, mask=None):
        # Pre-LN på Attention grenen
        norm_x = self.norm1(x, ret)
        attn_out = self.self_attn(norm_x, mask)
        x = x + self.dropout1(attn_out)

        # Pre-LN på FeedForward grenen
        norm_x2 = self.norm2(x, ret)
        ff_out = self.ffn(norm_x2)
        x = x + self.dropout2(ff_out)

        return x

class TransformerEncoderLayer(nn.Module):
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

if __name__ == '__main__':
    # def __init__(self, d_model, num_layers, num_heads, max_len, num_bins, dropout = 0.1)
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('./config.yaml')
    '''predictor = Predictor(
        d_model = cfg['jepa']['d_model'],
        num_layers = cfg['jepa']['predictor']['num_layers'],
        num_heads = cfg['jepa']['predictor']['num_heads'],
        max_len = cfg['jepa']['predictor']['max_len'],
        num_bins = cfg['jepa']['predictor']['num_bins']
    )

    Z = torch.rand((cfg['jepa']['training']['batch_size'], cfg['jepa']['predictor']['max_len'], cfg['jepa']['d_model']))
    Z_prime = predictor(Z)'''
    from data import CryptoDataset
    from util import set_logger
    logger = set_logger(cfg)
    data = CryptoDataset(cfg)
    t = data[0]['target']
    print(t.shape)
    print(t)