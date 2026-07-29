from torchvision import datasets


# ============================================================
# CIFAR-10 SimCLR Dataset
# Returns two independently augmented views of each image.
# ============================================================

class CIFAR10SimCLR(datasets.CIFAR10):
    """
    CIFAR-10 dataset wrapper for SimCLR contrastive learning.

    Each sample returns two independently augmented views
    generated from the same original image.
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        download=False,
    ):
        super().__init__(
            root=root,
            train=train,
            transform=None,
            download=download,
        )

        self.simclr_transform = transform

    def __getitem__(self, index):
        """
        Return two augmented views and the original label.

        Args:
            index (int): Dataset sample index.

        Returns:
            tuple:
                view_1: First augmented view.
                view_2: Second augmented view.
                label: Original CIFAR-10 label.
        """

        image, label = self.data[index], self.targets[index]

        image = self._to_pil_image(image)

        view_1 = self.simclr_transform(image)
        view_2 = self.simclr_transform(image)

        return view_1, view_2, label

    @staticmethod
    def _to_pil_image(image):
        """
        Convert a NumPy image array to a PIL image.
        """

        from PIL import Image

        return Image.fromarray(image)