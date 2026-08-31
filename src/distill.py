"""
Implements DeiT's "hard-label distillation" loss.

The idea (Touvron et al., 2021): the student has two output heads (cls and
distillation). The cls head is trained normally on the true label. The
distillation head is trained on the TEACHER's predicted class (its argmax,
i.e. a "hard" pseudo-label) rather than the teacher's soft probabilities -
the paper found hard labels work at least as well and need no extra
temperature hyperparameter. Final loss is a simple average of the two.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, teacher, alpha: float = 0.5):
        super().__init__()
        self.teacher = teacher
        self.alpha = alpha
        self.base_criterion = nn.CrossEntropyLoss()

    def forward(self, images, outputs, labels):
        """
        images:  the same batch of input images that produced `outputs`
                 (the teacher needs to see them too, to form its own opinion).
        outputs: tuple (cls_logits, dist_logits) from the DeiT student -
                 this is what the model returns automatically while .train().
        labels:  true labels for this batch.
        """
        cls_logits, dist_logits = outputs

        # Head 1: normal supervised loss against the true label
        base_loss = self.base_criterion(cls_logits, labels)

        # Head 2: distillation loss against the teacher's hard pseudo-label
        with torch.no_grad():
            teacher_logits = self.teacher(images)
            teacher_labels = teacher_logits.argmax(dim=1)
        distill_loss = F.cross_entropy(dist_logits, teacher_labels)

        return (1 - self.alpha) * base_loss + self.alpha * distill_loss