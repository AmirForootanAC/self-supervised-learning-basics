import os

import torch
import matplotlib.pyplot as plt

from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.data.transforms import create_validation_transform
from src.models.resnet_classifier import ResNet18Classifier
from src.models.simclr_model import SimCLR
from src.models.linear_classifier import LinearClassifier


# ============================================================
# Configuration
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

TEST_BATCH_SIZE = 128

BASELINE_CHECKPOINT = (
    "results/best_baseline.pth"
)

FINETUNE_CHECKPOINT = (
    "results/best_finetune.pth"
)

RESULTS_DIR = "results"


# CIFAR-10 class names
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# ============================================================
# Load Baseline Model
# ============================================================


def load_baseline_model():

    model = ResNet18Classifier()

    checkpoint = torch.load(
        BASELINE_CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint
    )

    model = model.to(DEVICE)

    return model


# ============================================================
# Load Fine-tuned SimCLR Model
# ============================================================


def load_finetuned_model():

    simclr_model = SimCLR()

    classifier = LinearClassifier()

    checkpoint = torch.load(
        FINETUNE_CHECKPOINT,
        map_location=DEVICE,
    )

    simclr_model.encoder.load_state_dict(
        checkpoint[
            "encoder_state_dict"
        ]
    )

    classifier.load_state_dict(
        checkpoint[
            "classifier_state_dict"
        ]
    )

    encoder = simclr_model.encoder

    encoder = encoder.to(DEVICE)

    classifier = classifier.to(DEVICE)

    return encoder, classifier


# ============================================================
# Get Baseline Predictions
# ============================================================


def get_baseline_predictions(
    model,
    dataloader,
):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    return (
        all_labels,
        all_predictions,
    )


# ============================================================
# Get Fine-tuned SimCLR Predictions
# ============================================================


def get_finetune_predictions(
    encoder,
    classifier,
    dataloader,
):

    encoder.eval()
    classifier.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(DEVICE)

            features = encoder(
                images
            )

            outputs = classifier(
                features
            )

            predictions = outputs.argmax(
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    return (
        all_labels,
        all_predictions,
    )


# ============================================================
# Save Confusion Matrix
# ============================================================


def save_confusion_matrix(
    labels,
    predictions,
    title,
    filename,
):

    cm = confusion_matrix(
        labels,
        predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
    )

    ax.set_title(
        title
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        filename,
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Main
# ============================================================


def main():

    print(
        "Device:",
        DEVICE,
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Test Dataset
    # --------------------------------------------------------

    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        transform=create_validation_transform(),
        download=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
    )


    print(
        "Test samples:",
        len(test_dataset),
    )


    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    print(
        "\nLoading Supervised Baseline..."
    )

    baseline_model = (
        load_baseline_model()
    )

    baseline_labels, baseline_predictions = (
        get_baseline_predictions(
            baseline_model,
            test_loader,
        )
    )


    save_confusion_matrix(
        baseline_labels,
        baseline_predictions,
        "Supervised Baseline - CIFAR-10 Test Set",
        "confusion_matrix_baseline.png",
    )


    # --------------------------------------------------------
    # SimCLR + Fine-tuning
    # --------------------------------------------------------

    print(
        "\nLoading SimCLR + Fine-tuning..."
    )

    encoder, classifier = (
        load_finetuned_model()
    )

    finetune_labels, finetune_predictions = (
        get_finetune_predictions(
            encoder,
            classifier,
            test_loader,
        )
    )


    save_confusion_matrix(
        finetune_labels,
        finetune_predictions,
        "SimCLR + Fine-tuning - CIFAR-10 Test Set",
        "confusion_matrix_finetune.png",
    )


    print(
        "\nConfusion matrix generation completed."
    )


# ============================================================
# Entry Point
# ============================================================


if __name__ == "__main__":

    main()
