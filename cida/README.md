
# CIDA: Causal & Style-Disentangled Image Training (Modular)

This repository provides a **modular** implementation of CIDA with **ResNet50 as the default backbone**,
**dual-branch heads** (`zc_head`, `zs_head`), and optional **regularizers**.
A backbone zoo (via `timm`) is also supported for baseline comparisons.

## Key Features
- **Backbone**: Default `resnet50`; switchable via `--arch` to test baselines.
- **Dual Heads**: `zc_head` (causal) and `zs_head` (style) with multiple variants.
- **Regularizers**: HSIC, MMD, orthogonality, cross-covariance (configurable weights).
- **Counterfactual Aug**: Fourier low-frequency style mixing (optional).
- **Training**: AMP, AdamW, metrics logging, best checkpoint export.

## Install
```bash
pip install -e .
```

## Quickstart
```bash
cida   --data_root /path/to/data --num_classes 3   --zc_head mlp --zs_head mlp --zc_dim 256 --zs_dim 256   --epochs 50 --batch_size 32
```

## Baselines (via `--arch`)
- `resnet50` (default)
- `convnext_*` (e.g., `convnext_tiny`), `xcit_*` (e.g., `xcit_small_12_p16_224`),
  `twins_svt_*` (e.g., `twins_svt_small`), `davit_*` (e.g., `davit_tiny`),
  `beit_*` (e.g., `beit_base_patch16_224`), and—if your `timm` version supports—
  `repvit_*`, `fastvit_*`.
- Classic baseline: `vgg19` (falls back to torchvision if `timm` name unsupported).

> Availability depends on your installed **timm** version.

## Dataset structure
```
/root/
  train/classA/*.jpg
  train/classB/*.jpg
  val/classA/*.jpg
  val/classB/*.jpg
  test/... (optional)
```

## Outputs
- `checkpoints/last.pth`, `checkpoints/best.pth`
- `metrics_epoch.csv`, `final_metrics.json`

## CLI (full)
Run `cida --help` for all arguments, including regularizer weights:
- `--lambda_hsic`, `--lambda_mmd`, `--lambda_orth`, `--lambda_xcov`
- `--use_z_both 1` to classify on `[Zc, Zs]` instead of `Zc` only
- `--use_fourier 1` to enable Fourier style mixing

## Citation
```bibtex
@software{cida,
  title        = {CIDA: Causal & Style-Disentangled Image Training},
  author       = {Dr.},
  year         = 2025,
  url          = {https://github.com/<yourname>/cida}
}
```
