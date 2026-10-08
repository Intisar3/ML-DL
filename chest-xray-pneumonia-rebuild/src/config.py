"""All settings for the rebuild live here, so every run is reproducible."""
from dataclasses import dataclass


@dataclass
class Config:
    img_size: int = 224
    batch_size: int = 32
    seed: int = 42
    val_fraction: float = 0.125      # share of train+val patients held out for validation

    weights: str | None = "imagenet"  # None trains from scratch (used by the smoke test)
    dropout: float = 0.3

    head_epochs: int = 5             # phase 1: backbone frozen, train the new head
    head_lr: float = 1e-3
    finetune_epochs: int = 10        # phase 2: unfreeze the top of the backbone
    finetune_lr: float = 1e-5
    unfreeze_layers: int = 40        # last N backbone layers (BatchNorm stays frozen)
    patience: int = 3

    mixed_precision: bool = True     # applied only when a GPU is present
    cache: bool = True               # keep decoded, resized images in RAM (about 1 GB at 224 px)


# Each task: which backbone it uses and what its classes are (index = label id).
TASKS = {
    "stage1": {"backbone": "resnet50", "classes": ["normal", "pneumonia"]},
    "stage2": {"backbone": "inceptionv3", "classes": ["bacteria", "virus"]},
    "three_class": {"backbone": "resnet50", "classes": ["normal", "bacteria", "virus"]},
}
