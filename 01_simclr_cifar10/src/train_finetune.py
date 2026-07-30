import torch
import torch.nn as nn
from torch.optim import SGD
from tqdm import tqdm

from src.data.supervised_dataset import create_supervised_datasets
from src.data.transforms import (
    create_supervised_transform,
    create_validation_transform,
)

from src.models.simclr_model import SimCLR
from src.models.linear_classifier import LinearClassifier


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = "results/simclr/checkpoint.pth"


def load_pretrained_encoder():

    model = SimCLR()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    encoder = model.encoder

    encoder = encoder.to(DEVICE)

    return encoder


def train_epoch(
    encoder,
    classifier,
    loader,
    optimizer,
    criterion,
):

    encoder.train()
    classifier.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        features = encoder(images)

        outputs = classifier(features)

        loss = criterion(
            outputs,
            labels,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    accuracy = 100 * correct / total

    return (
        total_loss / len(loader),
        accuracy,
    )


def evaluate(
    encoder,
    classifier,
    loader,
):

    encoder.eval()
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            features = encoder(images)

            outputs = classifier(features)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return 100 * correct / total


def main():

    print("Device:", DEVICE)

    print(
        "Loading SimCLR pretrained encoder..."
    )

    encoder = load_pretrained_encoder()

    classifier = LinearClassifier().to(
        DEVICE
    )

    train_transform = (
        create_supervised_transform()
    )

    val_transform = (
        create_validation_transform()
    )

    train_dataset, val_dataset = (
        create_supervised_datasets(
            root="data",
            train_transform=train_transform,
            val_transform=val_transform,
        )
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = SGD(
        [
            {
                "params": encoder.parameters(),
                "lr": 0.01,
            },
            {
                "params": classifier.parameters(),
                "lr": 0.1,
            },
        ],
        momentum=0.9,
        weight_decay=1e-4,
    )

    epochs = 50

    best_val_acc = 0.0

    for epoch in range(epochs):

        loss, train_acc = train_epoch(
            encoder,
            classifier,
            train_loader,
            optimizer,
            criterion,
        )

        val_acc = evaluate(
            encoder,
            classifier,
            val_loader,
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                {
                    "encoder_state_dict":
                        encoder.state_dict(),

                    "classifier_state_dict":
                        classifier.state_dict(),

                    "epoch":
                        epoch + 1,

                    "val_accuracy":
                        val_acc,
                },
                "results/best_finetune.pth",
            )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Val Acc: {val_acc:.2f}%"
        )

    print(
        f"Best Fine-tune Val Accuracy: "
        f"{best_val_acc:.2f}%"
    )


if __name__ == "__main__":

    main()