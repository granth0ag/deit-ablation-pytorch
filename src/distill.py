"""
Hard-label knowledge distillation for DeiT.

The classification head is trained using the ground-truth labels,
while the distillation head is trained using the teacher's predicted
class. The two losses are combined using the distillation weight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, teacher, alpha=0.5):
        super().__init__()
        self.teacher = teacher
        self.alpha = alpha
        self.base_criterion = nn.CrossEntropyLoss()

    def forward(self, images, outputs, labels):
        cls_logits, dist_logits = outputs

        base_loss = self.base_criterion(cls_logits, labels)

        with torch.no_grad():
            teacher_logits = self.teacher(images)
            teacher_labels = teacher_logits.argmax(dim=1)

        distill_loss = F.cross_entropy(
            dist_logits,
            teacher_labels,
        )

        return (
            (1 - self.alpha) * base_loss
            + self.alpha * distill_loss
        )