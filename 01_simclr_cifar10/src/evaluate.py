
import torch
from torchvision import datasets
from torch.utils.data import DataLoader

from src.data.transforms import create_validation_transform
from src.models.resnet_classifier import ResNet18Classifier
from src.models.simclr_model import SimCLR
from src.models.linear_classifier import LinearClassifier


# ============================================================
# Model Evaluation
# Calculates classification accuracy.
# ============================================================


def evaluate(
    model,
    dataloader,
    device,
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1,
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = 100 * correct / total

    return accuracy


# ============================================================
# Final Test Evaluation
# Compares:
# 1. Supervised Baseline
# 2. SimCLR + Fine-tuning
#
# Both models are evaluated on the exact same
# CIFAR-10 test set.
# ============================================================


def load_baseline_model(
    checkpoint_path,
    device,
):

    model = ResNet18Classifier()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint
    )

    model = model.to(device)

    return model


def load_finetuned_model(
    checkpoint_path,
    device,
):

    encoder_model = SimCLR()

    classifier = LinearClassifier()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    encoder_model.encoder.load_state_dict(
        checkpoint["encoder_state_dict"]
    )

    classifier.load_state_dict(
        checkpoint["classifier_state_dict"]
    )

    encoder = encoder_model.encoder

    encoder = encoder.to(device)
    classifier = classifier.to(device)

    return encoder, classifier


def evaluate_finetuned_model(
    encoder,
    classifier,
    dataloader,
    device,
):

    encoder.eval()
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            features = encoder(images)

            outputs = classifier(features)

            _, predicted = torch.max(
                outputs,
                1,
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = 100 * correct / total

    return accuracy


def main():

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )


    # --------------------------------------------------------
    # CIFAR-10 Test Dataset
    # --------------------------------------------------------
    # Important:
    # train=False loads the official CIFAR-10 test set.
    # The test set is never used during training or
    # validation.
    # --------------------------------------------------------

    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        transform=create_validation_transform(),
        download=True,
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
    )


    print(
        "Test samples:",
        len(test_dataset),
    )


    # --------------------------------------------------------
    # 1. Supervised Baseline
    # --------------------------------------------------------

    print()
    print(
        "Evaluating Supervised Baseline..."
    )

    baseline_model = load_baseline_model(
        checkpoint_path="results/best_baseline.pth",
        device=device,
    )

    baseline_accuracy = evaluate(
        baseline_model,
        test_loader,
        device,
    )


    # --------------------------------------------------------
    # 2. SimCLR + Fine-tuning
    # --------------------------------------------------------

    print()
    print(
        "Evaluating SimCLR + Fine-tuning..."
    )

    encoder, classifier = load_finetuned_model(
        checkpoint_path="results/best_finetune.pth",
        device=device,
    )

    finetune_accuracy = evaluate_finetuned_model(
        encoder,
        classifier,
        test_loader,
        device,
    )


    # --------------------------------------------------------
    # Final Comparison
    # --------------------------------------------------------

    improvement = (
        finetune_accuracy
        - baseline_accuracy
    )


    print()
    print("=" * 50)
    print(
        "Final Test Evaluation"
    )
    print("=" * 50)

    print(
        f"Supervised Baseline       : "
        f"{baseline_accuracy:.2f}%"
    )

    print(
        f"SimCLR + Fine-tuning      : "
        f"{finetune_accuracy:.2f}%"
    )

    print(
        f"Improvement               : "
        f"{improvement:+.2f}%"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()
