
import torch
import torch.nn as nn

def _try_import_timm():
    try:
        import timm
        return timm
    except Exception:
        return None

def _try_import_tv():
    try:
        from torchvision import models as tvm
        return tvm
    except Exception:
        return None

class BackboneFeatures(nn.Module):
    def __init__(self, arch: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.arch = arch
        self.timm = _try_import_timm()
        self.tvm = _try_import_tv()
        self.pretrained = pretrained
        self._build()

    def _build(self):
        if self.timm is not None:
            try:
                self.model = self.timm.create_model(self.arch, pretrained=self.pretrained, num_classes=0, global_pool="avg")
                self.out_dim = self.model.num_features
                self.forward = self.forward_timm
                return
            except Exception:
                pass
        if self.tvm is not None and self.arch.lower() == "resnet50":
            m = self.tvm.resnet50(weights=self.tvm.ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None)
            layers = list(m.children())[:-1]
            self.model = nn.Sequential(*layers)  # [B, 2048, 1, 1]
            self.out_dim = 2048
            self.forward = self.forward_torchvision
            return
        if self.tvm is not None and self.arch.lower() == "vgg19":
            m = self.tvm.vgg19(weights=self.tvm.VGG19_Weights.IMAGENET1K_V1 if self.pretrained else None)
            self.model = nn.Sequential(m.features, nn.AdaptiveAvgPool2d((1,1)))
            self.out_dim = 512
            self.forward = self.forward_torchvision
            return
        raise ValueError(f"Unknown or unsupported architecture '{self.arch}'.")

    def forward_timm(self, x):
        return self.model(x)  # [B, C]

    def forward_torchvision(self, x):
        x = self.model(x)
        return torch.flatten(x, 1)
