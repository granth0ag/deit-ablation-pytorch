"""
Builds train/val dataloaders. Defaults to CIFAR-100 (auto-downloads) since
it's small enough to iterate on quickly; switch dataset: imagefolder in the
config to point at your own data/train and data/val folders instead.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.data.transforms import get_transforms


def build_dataloaders(cfg):
    train_tf, val_tf = get_transforms(cfg["augmentation"], cfg["image_size"])

    if cfg["dataset"] == "cifar100":
        train_set = datasets.CIFAR100(
            root=cfg["data_dir"], train=True, download=True, transform=train_tf
        )
        val_set = datasets.CIFAR100(
            root=cfg["data_dir"], train=False, download=True, transform=val_tf
        )

    elif cfg["dataset"] == "imagefolder":
        train_set = datasets.ImageFolder(f"{cfg['data_dir']}/train", transform=train_tf)
        val_set = datasets.ImageFolder(f"{cfg['data_dir']}/val", transform=val_tf)

    else:
        raise ValueError(f"Unknown dataset: {cfg['dataset']}")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,   # mixup needs even batch sizes; drop the ragged last batch
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader