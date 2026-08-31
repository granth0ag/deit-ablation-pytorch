import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Tracks a running average of a metric (loss, accuracy, ...) over a run."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Top-1 accuracy. Handles both hard integer labels and soft/one-hot labels
    (which show up when mixup/cutmix is turned on) by taking argmax either way.
    """
    preds = logits.argmax(dim=1)
    if targets.dim() > 1:  # soft labels from mixup -> convert to hard for reporting
        targets = targets.argmax(dim=1)
    return (preds == targets).float().mean().item()


def save_checkpoint(model, optimizer, epoch, cfg, filename="last.pt"):
    ckpt_dir = os.path.join(cfg["output_dir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
        },
        os.path.join(ckpt_dir, filename),
    )