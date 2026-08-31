"""
Single entrypoint for all three experiments. Which experiment runs is decided
entirely by which config file you pass in - this script never changes:

  python -m src.train --config configs/vanilla_vit.yaml
  python -m src.train --config configs/vit_deit_aug.yaml
  python -m src.train --config configs/deit_full.yaml
"""

import argparse
import os

import torch
import torch.nn as nn
from timm.loss import SoftTargetCrossEntropy
from tqdm import tqdm
import yaml

from src.data.dataset import build_dataloaders
from src.data.transforms import get_mixup_fn
from src.distill import DistillationLoss
from src.models.deit import build_deit, build_teacher
from src.models.vit import build_vit
from src.utils import AverageMeter, accuracy, save_checkpoint, set_seed


def build_model(cfg):
    if cfg["model"] == "vit":
        return build_vit(cfg)
    elif cfg["model"] == "deit":
        return build_deit(cfg)
    else:
        raise ValueError(f"Unknown model type: {cfg['model']}")


def build_criterion(cfg, teacher=None):
    """
    Picks the right loss function for this experiment:
      - distillation on  -> DistillationLoss (needs the teacher)
      - mixup on, no distillation -> SoftTargetCrossEntropy (labels are soft)
      - neither -> plain CrossEntropyLoss
    """
    if cfg["distillation"]:
        return DistillationLoss(teacher, alpha=cfg.get("distillation_alpha", 0.5))
    elif cfg.get("mixup", False):
        return SoftTargetCrossEntropy()
    else:
        return nn.CrossEntropyLoss()


def train_one_epoch(model, loader, optimizer, criterion, mixup_fn, cfg, device, epoch):
    model.train()
    loss_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]")
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        outputs = model(images)

        if cfg["distillation"]:
            loss = criterion(images, outputs, targets)
        else:
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=images.size(0))
        pbar.set_postfix(loss=loss_meter.avg)

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, device, epoch):
    model.eval()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]")
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)  # eval mode -> single tensor even for DeiT
        acc_meter.update(accuracy(outputs, targets), n=images.size(0))
        pbar.set_postfix(acc=acc_meter.avg)

    return acc_meter.avg


def main(cfg):
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = build_dataloaders(cfg)
    mixup_fn = get_mixup_fn(cfg)

    model = build_model(cfg).to(device)
    teacher = build_teacher(cfg).to(device) if cfg["distillation"] else None
    criterion = build_criterion(cfg, teacher)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_acc = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, mixup_fn, cfg, device, epoch
        )
        val_acc = validate(model, val_loader, device, epoch)
        scheduler.step()

        print(f"[{cfg['experiment_name']}] epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

        save_checkpoint(model, optimizer, epoch, cfg, filename="last.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, cfg, filename="best.pt")

    print(f"Done. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)