
import torch
import torch.nn as nn
from .backbone import BackboneFeatures
from .heads import create_zc_head, create_zs_head, LowRankHead

class CIDAModel(nn.Module):
    def __init__(self, arch: str, zc_head: str, zs_head: str, num_classes: int,
                 zc_dim: int = 256, zs_dim: int = 256, pretrained: bool = True,
                 use_z_both: bool = False, zc_rank: int = 64, style_code_dim: int = 128):
        super().__init__()
        self.backbone = BackboneFeatures(arch=arch, pretrained=pretrained)
        self.zc = create_zc_head(zc_head, self.backbone.out_dim, zc_dim, zc_rank=zc_rank)
        self.zs = create_zs_head(zs_head, self.backbone.out_dim, zs_dim, style_code_dim=style_code_dim)
        self.use_z_both = use_z_both
        cls_in = zc_dim + zs_dim if use_z_both else zc_dim
        self.classifier = nn.Linear(cls_in, num_classes)

    def get_lowrank_orth(self):
        if isinstance(self.zc, LowRankHead):
            return self.zc.orth_penalty()
        return torch.tensor(0.0, device=self.classifier.weight.device)

    def forward(self, x):
        feat = self.backbone(x)            # [B, C]
        zc = self.zc(feat)                 # [B, zc_dim]
        zs = self.zs(feat)                 # [B, zs_dim]
        z = torch.cat([zc, zs], dim=1) if self.use_z_both else zc
        logits = self.classifier(z)
        return logits, zc, zs
