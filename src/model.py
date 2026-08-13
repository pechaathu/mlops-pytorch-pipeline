"""Model definitions for image classification."""

import torch.nn as nn
from torchvision import models


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """Build and return a ResNet-18 adapted for 32x32 CIFAR-10 images."""
    if architecture == "resnet18":
        model = models.resnet18(weights=None, num_classes=num_classes)

        # Replace the 7x7 stem with a 3x3 stride-1 conv for small 32x32 images.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove the early max-pool so we keep spatial detail.
        model.maxpool = nn.Identity()

        return model

    raise ValueError(f"Unsupported architecture: {architecture}")
