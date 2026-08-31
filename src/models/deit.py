"""
Full DeiT: a ViT with a second learnable "distillation token" added alongside
the usual [CLS] token. timm already implements this architecture (it's what
"*_distilled_*" model names refer to), so we don't need to reimplement the
token mechanics ourselves - we just need to know how it behaves:

  - During TRAINING, model(x) returns a TUPLE: (cls_logits, dist_logits)
      cls_logits  -> trained normally against the true label
      dist_logits -> trained against the TEACHER's prediction (see distill.py)
  - During EVAL, model(x) returns a single tensor: the average of both heads

This file just builds that model, plus the separate CNN teacher whose
predictions the distillation token learns to imitate.
"""

import timm


def build_deit(cfg):
    model = timm.create_model(
        cfg["model_name"],          # e.g. "deit_small_distilled_patch16_224"
        pretrained=False,
        num_classes=cfg["num_classes"],
    )
    return model


def build_teacher(cfg):
    """
    The teacher is a separately/pretrained CNN whose job is only to produce
    predictions for the student to imitate - it is never trained itself.
    """
    teacher = timm.create_model(
        cfg["teacher_model"],       # e.g. "regnety_160"
        pretrained=True,
        num_classes=cfg["num_classes"],
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher