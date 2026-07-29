import torchvision.transforms as transforms


# ============================================================
# SimCLR Data Augmentation
# Creates two different augmented views from the same image.
# ============================================================

def create_simclr_transform(image_size=32):
    """
    Create the augmentation pipeline used to generate
    augmented views for SimCLR contrastive learning.

    Args:
        image_size (int): Target image size.

    Returns:
        transforms.Compose: SimCLR augmentation pipeline.
    """

    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=image_size,
            scale=(0.2, 1.0),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1,
                )
            ],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])