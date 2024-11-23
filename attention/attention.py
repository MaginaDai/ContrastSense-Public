
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import split_last, merge_last, AverageMeter

class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, hidden_dim=72, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
        # return x


class MultiProjection(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        return q, k, v


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, hidden_dim=72, n_heads=4):
        super().__init__()
        self.proj_q = nn.Linear(hidden_dim, hidden_dim)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim)
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = n_heads

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        # scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        scores = q @ k.transpose(-2, -1)

        scores = F.softmax(scores, dim=-1)

        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, hidden_dim=72, hidden_ff=144):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_ff)
        self.fc2 = nn.Linear(hidden_ff, hidden_dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))
    

def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))