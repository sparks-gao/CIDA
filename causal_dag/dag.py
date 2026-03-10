import os
import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

# 引入 Scikit-learn 进行多分类评估
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize


# ----------------------------- Utilities ---------------------------------

def dag_constraint(W: torch.Tensor) -> torch.Tensor:
    """NOTEARS DAG constraint: h(W) = Tr(expm(W \circ W)) - d
    W: (d, d)
    returns scalar tensor
    """
    WW = W * W
    # torch.matrix_exp is available in recent PyTorch (>=1.8)
    try:
        expm = torch.matrix_exp(WW)
    except AttributeError:
        # fallback via series or CPU numpy (not implemented here). Require new torch.
        raise RuntimeError("torch.matrix_exp not available; upgrade PyTorch to >=1.8")
    h = torch.trace(expm) - W.shape[0]
    return h


class ResNetCausal(nn.Module):
    def __init__(self, num_classes: int, pretrained=True, proj_dim=128, use_projection=True, device='cpu'):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

        # Load ResNet50 and remove fc
        backbone = models.resnet50(pretrained=pretrained)
        modules = list(backbone.children())[:-1]  # remove fc
        self.backbone = nn.Sequential(*modules)
        self.backbone_out_dim = backbone.fc.in_features

        self.use_projection = use_projection
        if use_projection:
            self.proj = nn.Sequential(
                nn.Linear(self.backbone_out_dim, proj_dim),
                nn.BatchNorm1d(proj_dim),
                nn.ReLU(inplace=True)
            )
            self.z_dim = proj_dim
        else:
            self.proj = None
            self.z_dim = self.backbone_out_dim

        # Classifier (from Z to logits)
        self.classifier = nn.Linear(self.z_dim, self.num_classes)

        # Augmented adjacency matrix W of size (d+1, d+1). Last node is Y.
        d_aug = self.z_dim + 1
        W_init = 1e-2 * torch.randn(d_aug, d_aug)
        self.W = nn.Parameter(W_init)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        # x -> backbone -> GAP -> flatten -> projection -> Z
        feat = self.backbone(x)  # shape (B, C, 1, 1)
        feat = torch.flatten(feat, 1)
        if self.use_projection:
            z = self.proj(feat)
        else:
            z = feat
        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.forward_backbone(x)  # Z: (B, d)

        # --- Causal Attention Mechanism (方案二) ---
        d = self.z_dim

        # 1. 提取 Z -> Y 的边权重作为注意力分数 (Attention Scores)
        # W[:d, d] 是最后一列（指向Y）的前 d 行（来自Z_i）
        # strengths: (d,)
        attention_scores = torch.abs(self.W[:d, d])

        # 2. 归一化分数 (L1 归一化)
        # 确保数值稳定
        norm_factor = torch.sum(attention_scores) + 1e-6
        attention_weights = attention_scores / norm_factor  # attention_weights: (d,)

        # 3. 应用注意力：将 Z 按注意力权重重新加权
        # z: (B, d), attention_weights: (d,)
        z_focused = z * attention_weights

        # --- 分类器使用聚焦特征 ---

        # 替换 z 为 z_focused
        logit = self.classifier(z_focused)  # shape (B, num_classes)

        # 返回原始 Z 用于计算 Recon Loss (这是NOTEARS的要求)
        return logit, z

    # ----------------------------- Training loop -----------------------------


def train_epoch(model: ResNetCausal, dataloader: DataLoader, optimizer, device: str,
                alpha: float, gamma: float, lambda_W: float, epoch: int, verbose: bool = True):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_recon = 0.0
    total_h = 0.0

    for batch_idx, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = targets.to(device).long()

        # forward 返回 logit (用于 CE Loss) 和 原始 Z (用于 Recon Loss)
        logit, z = model(imgs)

        # Classification loss: CrossEntropyLoss
        ce_loss = F.cross_entropy(logit, targets)

        # Build augmented matrix Z_tilde = [Z, Y] for Causal Loss
        # 使用类别索引作为标量 Y 节点的值
        y_col = targets.unsqueeze(1).float()  # shape (B,1)
        z_tilde = torch.cat([z, y_col], dim=1)  # (B, d+1)

        # Reconstruction via W: Z_tilde @ W
        recon = z_tilde @ model.W
        recon_loss = F.mse_loss(recon, z_tilde)

        # DAG constraint
        h = dag_constraint(model.W)

        # Total Loss = CE + alpha * Recon + gamma * h(W)^2 + lambda_W * |W|_1
        loss = ce_loss + alpha * recon_loss + gamma * (h * h) + lambda_W * torch.sum(torch.abs(model.W))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_recon += recon_loss.item()
        total_h += h.item()

        if verbose and batch_idx % 50 == 0:
            print(
                f"Epoch {epoch} Batch {batch_idx}: loss={loss.item():.4f} ce={ce_loss.item():.4f} recon={recon_loss.item():.4f} h={h.item():.4f}")

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'ce': total_ce / n,
        'recon': total_recon / n,
        'h': total_h / n
    }


def evaluate(model: ResNetCausal, dataloader: DataLoader, device: str):
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            # forward now uses causal attention but returns raw Z
            logit, z = model(imgs)
            all_logits.append(logit.detach().cpu())
            all_targets.append(targets.detach().cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets).numpy()

    # 1. 准确率 (Accuracy)
    preds = torch.argmax(logits, dim=1).numpy()
    acc = accuracy_score(targets, preds)

    # 2. AUC (Area Under the Curve, Macro-AUC)
    probs = F.softmax(logits, dim=1).numpy()
    num_classes = model.num_classes

    macro_auc = 0.0

    # 检查是否有足够的类别进行 AUC 计算
    if num_classes > 1 and len(np.unique(targets)) > 1:
        try:
            # 对 targets 进行 One-Hot 编码
            targets_binarized = label_binarize(targets, classes=range(num_classes))
            # 确保 binarized targets 的维度正确 (N, num_classes)
            if targets_binarized.ndim == 1:
                targets_binarized = np.expand_dims(targets_binarized, axis=1)

            # 计算 Macro-AUC (One-vs-Rest, 宏平均)
            macro_auc = roc_auc_score(targets_binarized, probs, multi_class='ovr', average='macro')
        except ValueError as e:
            print(f"Warning: AUC calculation failed ({e}). Returning 0.0.")
            macro_auc = 0.0
    elif num_classes > 1:
        print("Warning: Only one class present in the evaluation set. AUC cannot be calculated.")

    return {'acc': acc, 'macro_auc': macro_auc}


# ----------------------------- Helpers ----------------------------------

def get_causal_strengths(model: ResNetCausal) -> torch.Tensor:
    """Return absolute weights from feature_i -> Y."""
    W = model.W.detach().cpu()
    d = model.z_dim
    # W[:d, d] 对应 Z_i -> Y 的边
    strengths = torch.abs(W[:d, d])
    return strengths


# ----------------------------- Example main -----------------------------

def build_dataloaders(data_dir: str, img_size: int, batch_size: int):
    # 根据路径结构 /data/ghl/lung_data_7.21/train 和 val
    data_path = os.path.join(data_dir, 'lung_data_7.21')

    # 图像预处理
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_path, 'val'), transform=val_tf)

    num_classes = len(train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, num_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/ghl/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1.0)  # weight for recon
    parser.add_argument('--gamma', type=float, default=0.1)  # weight for h(W)^2
    parser.add_argument('--lambda_W', type=float, default=1e-4)  # L1 on W
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--use_projection', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    # 加载数据并获取类别数
    train_loader, val_loader, num_classes = build_dataloaders(args.data_dir, args.img_size, args.batch_size)
    print(f"Dataset loaded. Number of classes: {num_classes}")
    if num_classes < 2:
        print("Error: Need at least 2 classes for classification.")
        return

    # 模型初始化
    model = ResNetCausal(num_classes=num_classes, pretrained=True, proj_dim=args.proj_dim,
                         use_projection=args.use_projection, device=device)
    model.to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print("Starting training with Causal Attention...")
    for epoch in range(1, args.epochs + 1):
        stats = train_epoch(model, train_loader, optimizer, device, args.alpha, args.gamma, args.lambda_W, epoch)

        # 评估
        val_stats = evaluate(model, val_loader, device)

        # 打印结果，包括 Acc 和 Macro-AUC
        print(
            f"Epoch {epoch} TRAIN loss={stats['loss']:.4f} ce={stats['ce']:.4f} recon={stats['recon']:.4f} h={stats['h']:.6f} | VAL Acc={val_stats['acc']:.4f} AUC={val_stats['macro_auc']:.4f}")

        # 周期性：检查因果强度
        strengths = get_causal_strengths(model)
        topk = min(10, strengths.shape[0])
        # 打印 Top K 具有最高因果强度的特征索引
        if strengths.numel() > 0:
            topk_values, topk_idx = torch.topk(strengths, topk)
            print(
                f"Top-{topk} feature indices (strongest -> Y): {topk_idx.numpy().tolist()} (Values: {topk_values.numpy()})")

        # 保存模型
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, f'model_epoch_{epoch}.pth')
    print("Training finished.")


if __name__ == '__main__':
    main()