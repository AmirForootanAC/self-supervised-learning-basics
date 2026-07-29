from torch.utils.data import DataLoader


from .simclr_dataset import SimCLRDataset



# ============================================================
# Create SimCLR DataLoader
#
# Returns:
#   view_1 batch
#   view_2 batch
#
# Labels are not used.
#
# ============================================================


def create_simclr_dataloader(
    root,
    transform,
    batch_size=128,
    train=True,
    download=False,
    num_workers=2,
):


    dataset = SimCLRDataset(
        root=root,
        transform=transform,
        train=train,
        download=download,
    )


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )


    return dataloader