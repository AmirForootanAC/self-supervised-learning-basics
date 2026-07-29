import torchvision.transforms as transforms


# ============================================================
# CIFAR-10 Supervised Training Transform
# Basic augmentation for classification baseline.
# ============================================================


def create_supervised_transform():
    """
    Create transform for supervised CIFAR-10 training.
    """

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),

            transforms.RandomCrop(
                32,
                padding=4,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.4914,
                    0.4822,
                    0.4465,
                ],
                std=[
                    0.2470,
                    0.2435,
                    0.2616,
                ],
            ),
        ]
    )

    return transform



# ============================================================
# SimCLR Augmentation Transform
# Creates two random views from the same image.
# ============================================================

def create_simclr_transform():

    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=32,
                scale=(0.2, 1.0),
            ),

            transforms.RandomHorizontalFlip(
                p=0.5,
            ),

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

            transforms.RandomGrayscale(
                p=0.2,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.4914,
                    0.4822,
                    0.4465,
                ],
                std=[
                    0.2470,
                    0.2435,
                    0.2616,
                ],
            ),
        ]
    )

    return transform



# ============================================================
# CIFAR-10 Validation Transform
# No random augmentation.
# ============================================================


def create_validation_transform():
    """
    Create validation transform.
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.4914,
                    0.4822,
                    0.4465,
                ],
                std=[
                    0.2470,
                    0.2435,
                    0.2616,
                ],
            ),
        ]
    )

    return transform