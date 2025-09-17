# CIDA
CAUSAL SUBSPACE LEARNING WITH COUNTERFACTUAL REGULARIZATION FOR MEDICAL IMAGE ANALYSIS

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




# Requirements:
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.12
scikit-learn>=1.2.0
numpy>=1.23.0
tqdm>=4.62.0
pandas>=1.5.0

# Public Dataset:
LIDC-IDRI: https://cancerimagingarchive.net/collection/lidc-idri/
MedMNIST: https://zenodo.org/records/10519652
@article{medmnistv2,
    title={MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
    author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
    journal={Scientific Data},
    volume={10},
    number={1},
    pages={41},
    year={2023},
    publisher={Nature Publishing Group UK London}
}
