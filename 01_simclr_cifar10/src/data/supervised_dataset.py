from torchvision import datasets
from torch.utils.data import random_split


# ============================================================
# CIFAR-10 Supervised Dataset
# Creates train and validation splits.
# ============================================================


def create_supervised_datasets(
    root,
    train_transform=None,
    val_transform=None,
):
    """
    Create CIFAR-10 train and validation datasets.

    Split:
        Train: 45000
        Validation: 5000
    """


    full_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        transform=train_transform,
        download=False,
    )


    train_size = 45000
    val_size = 5000


    train_dataset, val_dataset = random_split(
        full_dataset,
        [
            train_size,
            val_size,
        ],
    )


    # Validation should not use training augmentation
    val_dataset.dataset = datasets.CIFAR10(
        root=root,
        train=True,
        transform=val_transform,
        download=False,
    )


    return train_dataset, val_dataset