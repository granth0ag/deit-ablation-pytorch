"""
Plain Vision Transformer, no distillation token. Used by experiments 1 and 2 -
the only thing that changes between those two is the data augmentation, not
the model architecture.
"""

import timm


def build_vit(cfg):
    model = timm.create_model(
        cfg["model_name"],          # e.g. "vit_small_patch16_224"
        pretrained=False,
        num_classes=cfg["num_classes"],
    )
    return model