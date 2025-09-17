
import os, json, csv, random, math
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from ..models.model import CIDAModel
from ..regularizers.losses import CELoss, hsic_loss, mmd_loss, orthogonality_reg, cross_covariance_reg
from ..utils.fourier import low_freq_mix

def seed_all(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_dataloaders(root: str, batch_size: int, num_workers: int = 8, img_size: int = 192):
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    train_set = datasets.ImageFolder(os.path.join(root, "train"), tf_train)
    val_set = datasets.ImageFolder(os.path.join(root, "val"), tf_eval)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, len(train_set.classes)

def cosine_sched(start, end, cur):
    return 0.5 * (1 - math.cos(math.pi * (cur - start) / max(1, end - start)))

def info_nce(z1, z2, t=0.2, maxk=None):
    # normalized projection for contrastive
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    if maxk is not None and min(z1.size(0), z2.size(0)) > maxk:
        idx = torch.randperm(z1.size(0), device=z1.device)[:maxk]
        z1 = z1[idx]; z2 = z2[idx]
    logits = z1 @ z2.t() / max(t, 1e-6)
    labels = torch.arange(z1.size(0), device=z1.device)
    return nn.CrossEntropyLoss()(logits, labels)

def jacobian_penalty(model, x, eps=1e-3, maxk=None):
    # Simple finite-diff approx on zc branch for small subset
    x.requires_grad_(True)
    logits, zc, zs = model(x)
    if maxk is not None and x.size(0) > maxk:
        idx = torch.randperm(x.size(0), device=x.device)[:maxk]
        zc = zc[idx]; x = x[idx]
    g = torch.autograd.grad(zc.sum(), x, retain_graph=True, create_graph=True)[0]
    return (g.pow(2)).mean()

def train_one_epoch(model, loader, optimizer, scaler, device, ce_loss, epoch, args, alt_toggle):
    model.train()
    total_loss = 0.0
    preds, gts = [], []
    step = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Build counterfactual ~x by low-frequency style mix unless disabled
        use_cf = not args.no_counterfactual
        if use_cf:
            idx = torch.randperm(imgs.size(0), device=device)
            imgs_cf = low_freq_mix(imgs, imgs[idx], alpha=max(1e-4, args.lowfreq_ratio))
        else:
            imgs_cf = imgs

        # Alternate detach between branches if requested
        detach_x = False; detach_cf = False
        if args.alt_detach and use_cf:
            detach_x = bool(alt_toggle[0] % 2 == 0)
            detach_cf = not detach_x
            alt_toggle[0] += 1

        # Gradient accumulation
        accum = max(1, args.accum_steps)
        optimizer.zero_grad(set_to_none=True)

        for micro in range(accum):
            # In accum mode, we split the batch virtually; here we just scale loss
            with torch.cuda.amp.autocast(enabled=not args.no_amp and scaler.is_enabled()):
                logits_x, zc_x, zs_x = model(imgs.detach() if detach_x else imgs)
                logits_cf, zc_cf, zs_cf = model(imgs_cf.detach() if detach_cf else imgs_cf)

                # CE on factual x
                loss = ce_loss(logits_x, labels)
                # CIC: encourage invariance between zc_x and zc_cf (InfoNCE/MSE proxy)
                if args.cic_start <= epoch <= args.cic_end and args.lambda_cic > 0:
                    if args.gamma_info > 0:
                        loss = loss + args.lambda_cic * info_nce(zc_x, zc_cf, t=args.infonce_t, maxk=args.infonce_maxk)
                    else:
                        loss = loss + args.lambda_cic * torch.nn.functional.mse_loss(zc_x, zc_cf)

                # CMDA: encourage domain/style alignment on Zs (MMD)
                if args.cmda_start <= epoch <= args.cmda_end and args.lambda_cmda > 0:
                    loss = loss + args.lambda_cmda * mmd_loss(zs_x, zs_cf)

                # CIR: independence/regularization between Zc and Zs
                if args.cir_start <= epoch <= args.cir_end and args.lambda_cir > 0:
                    reg_orth = orthogonality_reg(zc_x, zs_x)
                    reg_xcov = cross_covariance_reg(zc_x, zs_x)
                    reg_hsic = hsic_loss(zc_x, zs_x)
                    cir = args.alpha * reg_orth + (1 - args.alpha) * reg_xcov + args.beta * reg_hsic
                    loss = loss + args.lambda_cir * cir

                # Extra: style magnitude penalty on Zs (L2)
                if args.lambda_style > 0:
                    loss = loss + args.lambda_style * (zs_x.pow(2).mean())

                # Extra: explicit cross-cov reg (already in CIR but also separately weighted)
                if args.lambda_xcov > 0:
                    loss = loss + args.lambda_xcov * cross_covariance_reg(zc_x, zs_x)

                # Low-rank orth penalty if applicable
                if args.lambda_lowrank_orth > 0:
                    lowrank_orth = model.get_lowrank_orth()
                    loss = loss + args.lambda_lowrank_orth * lowrank_orth

                # Optional Jacobian penalty on Zc (small)
                if args.jacobian_penalty > 0:
                    jp = jacobian_penalty(model, imgs, maxk=min(args.infonce_maxk, 32))
                    loss = loss + args.jacobian_penalty * jp

                loss = loss / accum

            scaler.scale(loss).backward()

        scaler.step(optimizer); scaler.update()

        total_loss += float(loss.detach().item()) * imgs.size(0) * accum
        preds.extend(torch.argmax(logits_x, dim=1).detach().cpu().tolist())
        gts.extend(labels.detach().cpu().tolist())
        step += 1

    acc = accuracy_score(gts, preds)
    return total_loss / len(loader.dataset), acc

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gts, probs = [], [], []
    ce = nn.CrossEntropyLoss(reduction='sum'); total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, _, _ = model(imgs)
        total_loss += float(ce(logits, labels).item())
        p = torch.softmax(logits, dim=1)
        probs.extend(p.detach().cpu().tolist())
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        gts.extend(labels.detach().cpu().tolist())
    acc = accuracy_score(gts, preds); f1m = f1_score(gts, preds, average='macro')
    try: auroc = roc_auc_score(gts, probs, multi_class='ovr')
    except Exception: auroc = float('nan')
    cm = confusion_matrix(gts, preds).tolist()
    return total_loss / len(loader.dataset), acc, f1m, auroc, cm

def maybe_export_embeddings(model, loader, device, out_dir, epoch):
    import numpy as np
    model.eval()
    all_zc, all_zs, all_y = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            _, zc, zs = model(imgs)
        all_zc.append(zc.detach().cpu().numpy())
        all_zs.append(zs.detach().cpu().numpy())
        all_y.append(labels.numpy())
    all_zc = np.concatenate(all_zc, axis=0)
    all_zs = np.concatenate(all_zs, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"zc_epoch{epoch:03d}.npy"), all_zc)
    np.save(os.path.join(out_dir, f"zs_epoch{epoch:03d}.npy"), all_zs)
    np.save(os.path.join(out_dir, f"y_epoch{epoch:03d}.npy"), all_y)

def enable_grad_checkpointing(model):
    # best-effort: enable if the timm backbone has the method
    m = getattr(model.backbone, "model", None)
    if m is not None and hasattr(m, "set_grad_checkpointing"):
        try:
            m.set_grad_checkpointing(True)
            print("[info] Enabled gradient checkpointing in backbone (timm).")
        except Exception:
            pass

def main():
    p = ArgumentParser()
    p.add_argument('--data_root', type=str, required=True, help='/root/{train,val}/{class}/...')
    p.add_argument('--num_classes', type=int, default=3)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--accum_steps', type=int, default=2)
    p.add_argument('--img_size', type=int, default=192)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=5e-4)

    # backbone + heads
    p.add_argument('--arch', type=str, default='resnet50')
    p.add_argument('--zc_head', type=str, default='mlp', choices=['mlp','lowrank','grmlp'])
    p.add_argument('--zs_head', type=str, default='mlp', choices=['mlp','deep','film'])
    p.add_argument('--zc_rank', type=int, default=64, help='rank for lowrank Zc head')
    p.add_argument('--style_code_dim', type=int, default=128, help='FiLM style-code dim for Zs head')
    p.add_argument('--lambda_lowrank_orth', type=float, default=1e-4, help='orth reg for lowrank head')
    p.add_argument('--use_z_both', type=int, default=0)

    # objective weights
    p.add_argument('--lambda_cmda', type=float, default=0.5)
    p.add_argument('--lambda_cic', type=float, default=1.0)
    p.add_argument('--lambda_cir', type=float, default=0.5)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--temp', type=float, default=2.0)
    p.add_argument('--lowfreq_ratio', type=float, default=0.01)

    # extras
    p.add_argument('--gamma_info', type=float, default=0.2)
    p.add_argument('--lambda_xcov', type=float, default=0.1)
    p.add_argument('--lambda_style', type=float, default=0.02)
    p.add_argument('--infonce_t', type=float, default=0.2)
    p.add_argument('--jacobian_penalty', type=float, default=0.0)

    # subsampling caps
    p.add_argument('--mmd_maxk', type=int, default=128)
    p.add_argument('--hsic_maxk', type=int, default=128)
    p.add_argument('--infonce_maxk', type=int, default=128)

    # schedules
    p.add_argument('--cic_start', type=int, default=10)
    p.add_argument('--cic_end', type=int, default=30)
    p.add_argument('--cmda_start', type=int, default=30)
    p.add_argument('--cmda_end', type=int, default=100)
    p.add_argument('--cir_start', type=int, default=30)
    p.add_argument('--cir_end', type=int, default=100)

    # misc
    p.add_argument('--output', type=str, default='./checkpoints_icassp_heads')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--export_every', type=int, default=0, help='export Zc/Zs every N epochs (0=disable)')
    p.add_argument('--alt_detach', action='store_true', help='alternate detach between x/~x branches')
    p.add_argument('--no_counterfactual', action='store_true', help='disable ~x branch to debug memory')
    p.add_argument('--ckpt_backbone', action='store_true', help='enable gradient checkpointing in backbone')

    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, inferred = build_dataloaders(args.data_root, args.batch_size, args.num_workers, args.img_size)
    if inferred != args.num_classes:
        print(f"[Warn] Dataset classes={inferred}, but --num_classes={args.num_classes}. Using CLI value.")

    model = CIDAModel(arch=args.arch,
                      zc_head=args.zc_head, zs_head=args.zs_head,
                      num_classes=args.num_classes,
                      zc_dim=256, zs_dim=256,
                      pretrained=True,
                      use_z_both=bool(args.use_z_both),
                      zc_rank=args.zc_rank,
                      style_code_dim=args.style_code_dim).to(device)

    if args.ckpt_backbone:
        from ..models.backbone import BackboneFeatures
        try:
            # Enable timm checkpointing if available
            m = getattr(model.backbone, "model", None)
            if m is not None and hasattr(m, "set_grad_checkpointing"):
                m.set_grad_checkpointing(True)
                print("[info] Enabled gradient checkpointing for backbone.")
        except Exception:
            pass

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)

    ce_loss = CELoss()

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    metrics_csv = os.path.join(out_dir, "metrics_epoch.csv")
    with open(metrics_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","train_acc","val_loss","val_acc","val_f1","val_auroc"])

    best_acc = -1.0
    alt_toggle = [0]  # state for alternating detach
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, ce_loss, epoch, args, alt_toggle)
        val_loss, val_acc, val_f1, val_auroc, cm = evaluate(model, val_loader, device)

        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr_loss, tr_acc, val_loss, val_acc, val_f1, val_auroc])

        # Optional export of embeddings
        if args.export_every and (epoch % args.export_every == 0):
            maybe_export_embeddings(model, val_loader, device, os.path.join(out_dir, "embeddings"), epoch)

        is_best = val_acc > best_acc; best_acc = max(best_acc, val_acc)
        state = {"epoch": epoch, "arch": args.arch, "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "best_acc": best_acc}
        torch.save(state, os.path.join(out_dir, "last.pth"))
        if is_best: torch.save(state, os.path.join(out_dir, "best.pth"))

        print(f"Epoch {epoch:03d}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} auroc={val_auroc:.4f}")

    with open(os.path.join(out_dir, "final_metrics.json"), "w") as f:
        json.dump({"best_val_acc": best_acc}, f, indent=2)

if __name__ == '__main__':
    main()
