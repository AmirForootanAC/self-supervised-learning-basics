from torch.utils.data import DataLoader


# ============================================================
# DataLoader Factory
# Creates PyTorch DataLoaders for training and evaluation.
# ============================================================


def create_dataloader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
):
    """
    Create a PyTorch DataLoader.

    Args:
        dataset:
            PyTorch Dataset object.

        batch_size:
            Number of samples per batch.

        shuffle:
            Whether to shuffle dataset.

        num_workers:
            Number of parallel workers.

    Returns:
        DataLoader object.
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader