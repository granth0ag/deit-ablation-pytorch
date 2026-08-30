# ViT vs DeiT-Augmented vs Full DeiT

Three experiments, one codebase. Only the config file changes between runs.

| Experiment | Model | Augmentation | Distillation |
|---|---|---|---|
| `vanilla_vit.yaml` | plain ViT | standard | no |
| `vit_deit_aug.yaml` | plain ViT | DeiT-style (RandAugment, Mixup/CutMix, RandomErasing) | no |
| `deit_full.yaml` | DeiT (distillation token) | DeiT-style | yes, from a CNN teacher |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running an experiment

Run these from the **project root** (the folder this README is in), using `-m` so
the `src` package imports resolve correctly:

```bash
python -m src.train --config configs/vanilla_vit.yaml
python -m src.train --config