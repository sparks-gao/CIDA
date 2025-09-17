
import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, target):
        return self.ce(logits, target)

def rbf_kernel(x, y=None, gamma: float = None):
    if y is None: y = x
    if gamma is None: gamma = 1.0 / (x.shape[1] * x.var() + 1e-6)
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)
    return torch.exp(-gamma * (x_norm - 2.0 * x @ y.t() + y_norm.t()))

def mmd_loss(x, y, gamma: float = None):
    kxx = rbf_kernel(x, x, gamma); kyy = rbf_kernel(y, y, gamma); kxy = rbf_kernel(x, y, gamma)
    return kxx.mean() + kyy.mean() - 2 * kxy.mean()

def hsic_loss(x, y, gamma_x: float = None, gamma_y: float = None):
    K = rbf_kernel(x, gamma=gamma_x); L = rbf_kernel(y, gamma=gamma_y)
    n = K.size(0); H = torch.eye(n, device=K.device) - 1.0 / n
    KH = K @ H; LH = L @ H
    return (KH * LH).sum() / ((n - 1) ** 2)

def orthogonality_reg(zc, zs):
    zc = F.normalize(zc, dim=1); zs = F.normalize(zs, dim=1)
    return (zc @ zs.t()).pow(2).mean()

def cross_covariance_reg(zc, zs):
    zc = zc - zc.mean(dim=0, keepdim=True); zs = zs - zs.mean(dim=0, keepdim=True)
    cov = (zc.t() @ zs) / max(1, (zc.size(0) - 1))
    return cov.pow(2).mean()
