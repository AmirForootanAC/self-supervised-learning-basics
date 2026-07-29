import torch
from torch.optim import Adam


# ============================================================
# Baseline Training Runner
# Creates dataset, model, optimizer and starts training.
# ============================================================


from .data.supervised_dataset import create_supervised_datasets
from .data.transforms import (
    create_supervised_transform,
    create_validation_transform,
)

from .data.dataloader import create_dataloader

from .models.resnet_classifier import ResNet18Classifier

from .train_baseline import train_model



def main():

    # --------------------------------------------------------
    # Device configuration
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Device:",
        device
    )


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset, val_dataset = create_supervised_datasets(
        root="data",
        train_transform=create_supervised_transform(),
        val_transform=create_validation_transform(),
    )


    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )


    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = create_dataloader(
        train_dataset,
        batch_size=128,
        shuffle=True,
    )


    val_loader = create_dataloader(
        val_dataset,
        batch_size=128,
        shuffle=False,
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ResNet18Classifier()

    model.to(device)


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=1e-3,
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_accuracy = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=20,
    )


    print(
        f"Best Validation Accuracy: {best_accuracy:.2f}%"
    )



if __name__ == "__main__":
    main()