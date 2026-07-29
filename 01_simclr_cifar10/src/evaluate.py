import torch


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