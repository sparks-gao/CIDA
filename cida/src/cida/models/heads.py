
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class LowRankHead(nn.Module):
    """Linear low-rank factorization: W ≈ V @ U, rank=r.
    Also exposes an orthonormality penalty term for U.weight (rows)."""
    def __init__(self, in_dim, out_dim, rank=64):
        super().__init__()
        self.U = nn.Linear(in_dim, rank, bias=False)
        self.V = nn.Linear(rank, out_dim, bias=False)

    def orth_penalty(self):
        # Encourage U^T U ≈ I (row-orthogonality)
        W = self.U.weight  # [rank, in_dim]
        gram = W @ W.t()   # [rank, rank]
        I = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        return (gram - I).pow(2).mean()

    def forward(self, x):
        return self.V(self.U(x))

class GRMLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512, groups=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.gn = nn.GroupNorm(groups, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(self.gn(x.unsqueeze(-1)).squeeze(-1))
        x = self.fc2(x)
        return x

class DeepHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class FiLMHead(nn.Module):
    """A simple style-code head: first produces a style code s, then maps to output.
    This provides a 'style_code_dim' control to match the user's API."""
    def __init__(self, in_dim, out_dim, style_code_dim=128):
        super().__init__()
        self.s_proj = nn.Sequential(
            nn.Linear(in_dim, style_code_dim),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Linear(style_code_dim, out_dim)

    def forward(self, x):
        s = self.s_proj(x)
        y = self.out(s)
        return y

def create_zc_head(name: str, in_dim: int, out_dim: int, zc_rank: int = 64):
    name = name.lower()
    if name == "mlp": return MLPHead(in_dim, out_dim)
    if name == "lowrank": return LowRankHead(in_dim, out_dim, rank=zc_rank)
    if name == "grmlp": return GRMLPHead(in_dim, out_dim)
    raise ValueError(f"Unknown zc_head: {name}")

def create_zs_head(name: str, in_dim: int, out_dim: int, style_code_dim: int = 128):
    name = name.lower()
    if name == "mlp": return MLPHead(in_dim, out_dim)
    if name == "deep": return DeepHead(in_dim, out_dim)
    if name == "film": return FiLMHead(in_dim, out_dim, style_code_dim=style_code_dim)
    raise ValueError(f"Unknown zs_head: {name}")
