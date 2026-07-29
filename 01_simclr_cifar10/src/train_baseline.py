import torch
import torch.nn as nn

from tqdm import tqdm
import os


# ============================================================
# Supervised Baseline Training Loop
# Trains ResNet18 on CIFAR-10 labels.
# ============================================================


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
):
    """
    Train model for one epoch.
    """

    model.train()

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0


    for images, labels in tqdm(dataloader):

        images = images.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )


        loss.backward()

        optimizer.step()


        running_loss += loss.item()


        _, predicted = torch.max(
            outputs.data,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    epoch_loss = running_loss / len(dataloader)

    accuracy = 100 * correct / total


    return epoch_loss, accuracy

# ============================================================
# Multi Epoch Training
# ============================================================


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
):

    from .evaluate import evaluate


    best_accuracy = 0


    for epoch in range(epochs):

        loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )


        val_acc = evaluate(
            model,
            val_loader,
            device,
        )


        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Val Acc: {val_acc:.2f}%"
        )


        if val_acc > best_accuracy:

            best_accuracy = val_acc

            os.makedirs(
                "results",
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                "results/best_baseline.pth"
            )


    return best_accuracy