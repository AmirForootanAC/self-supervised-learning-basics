import os
import json

import torch
import torch.optim as optim
from tqdm import tqdm


from src.models.simclr_model import SimCLR
from src.losses.nt_xent import NTXentLoss

from src.data.simclr_dataloader import create_simclr_dataloader
from src.data.transforms import create_simclr_transform



# ============================================================
# Configuration
# ============================================================

EPOCHS = 50

BATCH_SIZE = 128

LEARNING_RATE = 3e-4

TEMPERATURE = 0.5


CHECKPOINT_PATH = "results/simclr/checkpoint.pth"

LOG_PATH = "results/simclr/train_log.json"

DATA_PATH = "data"



# ============================================================
# Checkpoint utility
# ============================================================


def checkpoint_exists():

    return os.path.exists(
        CHECKPOINT_PATH
    )



# ============================================================
# Training
# ============================================================


def train():

    if checkpoint_exists():

        print(
            "Checkpoint exists. Skipping training."
        )

        return



    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Device:",
        device
    )



    os.makedirs(
        "results/simclr",
        exist_ok=True
    )



    # -----------------------------
    # Dataset
    # -----------------------------

    loader = create_simclr_dataloader(
        root=DATA_PATH,
        transform=create_simclr_transform(),
        batch_size=BATCH_SIZE,
        train=True,
        download=False,
    )



    print(
        "Samples:",
        len(loader.dataset)
    )



    # -----------------------------
    # Model
    # -----------------------------

    model = SimCLR()

    model = model.to(device)



    # -----------------------------
    # Loss
    # -----------------------------

    criterion = NTXentLoss(
        temperature=TEMPERATURE
    )



    # -----------------------------
    # Optimizer
    # -----------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )



    history = []



    # -----------------------------
    # Training loop
    # -----------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0



        progress = tqdm(
            loader
        )


        for view1, view2 in progress:


            view1 = view1.to(device)

            view2 = view2.to(device)



            z1 = model(view1)

            z2 = model(view2)



            loss = criterion(
                z1,
                z2
            )



            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



            total_loss += loss.item()



            progress.set_description(
                f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss.item():.4f}"
            )



        avg_loss = total_loss / len(loader)



        print(
            f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}"
        )



        history.append(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
            }
        )



    # -----------------------------
    # Save checkpoint
    # -----------------------------

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": EPOCHS,
        },
        CHECKPOINT_PATH
    )



    with open(
        LOG_PATH,
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )



    print(
        "Training completed."
    )



if __name__ == "__main__":

    train()