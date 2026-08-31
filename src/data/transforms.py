"""
Defines two augmentation pipelines so experiments only differ by a config flag:

  "standard" -> what a plain ViT paper would use: resize/crop/flip/normalize
  "deit"     -> what the DeiT paper adds on top: RandAugment, RandomErasing,
                and (separately, applied at the BATCH level in train.py) Mixup/CutMix

Mixup/CutMix can't be a per-image transform (they mix two images together),
so they're built as a callable "collate-time" function here, and applied
inside the training loop in train.py — not inside these Compose pipelines.
"""

import torchvision.transforms as T
from timm.data import Mixup

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(augmentation: str, image_size: int = 224):
    """Returns (train_transform, val_transform) torchvision Compose objects."""

    val_transform = T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    if augmentation == "standard":
        train_transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    elif augmentation == "deit":
        train_transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.25),
        ])

    else:
        raise ValueError(f"Unknown augmentation type: {augmentation}")

    return train_transform, val_transform


def get_mixup_fn(cfg):
    """
    Returns a Mixup/CutMix function to call on each BATCH inside the training
    loop, or None if mixup is disabled in the config.

    Usage in train.py:
        images, targets = mixup_fn(images, targets)   # targets become soft labels
    """
    if not cfg.get("mixup", False):
        return None

    return Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.1,
        num_classes=cfg["num_classes"],
    )