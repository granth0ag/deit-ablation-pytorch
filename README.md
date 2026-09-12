# ViT vs DeiT-Augmented vs Full DeiT

A comparative study of vanilla ViT, ViT trained with DeiT-style augmentation, and full DeiT with knowledge distillation. All three experiments share the same training pipeline; the experimental setup is controlled entirely through YAML configuration files.

## Experiments

| Experiment | Model | Augmentation | Distillation |
|---|---|---|---|
| `vanilla_vit.yaml` | ViT | Standard | No |
| `vit_deit_aug.yaml` | ViT | DeiT-style | No |
| `deit_full.yaml` | DeiT | DeiT-style | Yes |

### 1. Vanilla ViT
A standard ViT baseline trained with conventional image augmentation (RandomResizedCrop, horizontal flip).

### 2. ViT + DeiT-style Augmentation
The same ViT architecture trained with DeiT-style augmentation:
- RandAugment
- Mixup / CutMix
- Random Erasing

This isolates the effect of stronger training augmentation without changing the model architecture.

### 3. Full DeiT
A DeiT model with:
- A distillation token, alongside DeiT-style augmentation
- Knowledge distillation from a pretrained CNN teacher

The teacher is a RegNetY-160 backbone pretrained on ImageNet, with its classification head reinitialized for CIFAR-100 and frozen during student training. **Note:** since the head is never fine-tuned on CIFAR-100, its predictions are currently a weak/noisy distillation signal — see [Known Limitations](#known-limitations).

## Implementation

Shared training pipeline for all three experimental conditions:
- ViT and DeiT model construction via `timm`
- DeiT-style data augmentation
- Mixup / CutMix
- Hard-label knowledge distillation
- Frozen pretrained CNN teacher
- AdamW optimizer with cosine LR scheduling
- YAML-based experiment configuration
- Checkpoint saving, validation, and accuracy tracking

The ViT/DeiT architectures themselves come from `timm`; the training loop, augmentation pipeline, and distillation loss are implemented in this repository.

## Experimental Setup

| Setting | Value |
|---|---|
| Dataset | CIFAR-100 |
| Image size | 224 × 224 |
| Batch size | 128 |
| Epochs | 30 |
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 0.05 |
| Seed | 42 |
| Teacher | RegNetY-160 |

## Running

Create the environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run an experiment from the project root, using `-m` so `src` package imports resolve correctly:

```bash
python -m src.train --config configs/vanilla_vit.yaml
python -m src.train --config configs/vit_deit_aug.yaml
python -m src.train --config configs/deit_full.yaml
```

## Project Structure

```
deit-ablation-pytorch/
├── configs/
│   ├── vanilla_vit.yaml
│   ├── vit_deit_aug.yaml
│   └── deit_full.yaml
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── vit.py
│   │   └── deit.py
│   ├── distill.py
│   ├── train.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```



## Known Limitations

- **Teacher is not fine-tuned on CIFAR-100.** `build_teacher` requests `num_classes=100` from a checkpoint pretrained on 1000-class ImageNet, which reinitializes the classifier head with random weights. That head is then frozen and never trained, so the "hard" pseudo-labels it produces are currently close to noise rather than a meaningful teaching signal. Fixing this (fine-tuning the teacher on CIFAR-100 first, or using a checkpoint already fine-tuned on it) is a prerequisite for the `deit_full` results to be meaningful.
- **No LR warmup**, despite training transformers from scratch — only cosine annealing is applied. Transformer training is known to be sensitive to this, especially without pretrained weights.
- **No mixed precision (AMP)** — training runs in fp32, which is slower and more memory-hungry than necessary on modern GPUs.
- Images are upsampled from CIFAR-100's native 32×32 to 224×224 to match the `patch16_224` architectures; since the models are trained from scratch (not loading ImageNet weights), this spends most of the compute upsampling low-resolution images rather than using a smaller patch/image size.

## Results

Full benchmark runs and quantitative analysis are currently pending. The training pipeline is implemented and runs end-to-end, but the three configurations have not yet been systematically benchmarked and compared.

Planned analysis:
- Validation accuracy comparison across the three configs
- Training loss comparison
- Effect of DeiT-style augmentation in isolation
- Effect of knowledge distillation (after addressing the teacher fine-tuning limitation above)

## Status

- [x] ViT training pipeline
- [x] DeiT-style augmentation pipeline
- [x] Knowledge distillation
- [x] CNN teacher integration
- [x] Config-based experiment setup
- [x] Checkpointing
- [ ] Teacher fine-tuning on CIFAR-100
- [ ] LR warmup
- [ ] Full benchmark runs
- [ ] Results analysis

## References

This project was developed by studying the DeiT paper (Touvron et al., 2021) and related open-source implementations. The `timm` library provides the ViT/DeiT model implementations.

## Notes

This repository focuses on understanding and comparing the training techniques introduced around DeiT, rather than reproducing the original paper's benchmark numbers.