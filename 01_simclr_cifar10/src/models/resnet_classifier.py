import torch.nn as nn
from torchvision.models import resnet18


# ============================================================
# ResNet18 CIFAR-10 Classifier
# Supervised baseline model.
# ============================================================


class ResNet18Classifier(nn.Module):
    """
    ResNet18 adapted for CIFAR-10 classification.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.model = resnet18(
            weights=None
        )

        # Adapt first convolution for CIFAR-10
        self.model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Remove ImageNet max pooling
        self.model.maxpool = nn.Identity()

        # Replace classifier head
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes,
        )

    def forward(self, x):
        return self.model(x)