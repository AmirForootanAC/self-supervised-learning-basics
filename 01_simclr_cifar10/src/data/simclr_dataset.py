import torch

from torch.utils.data import Dataset
from torchvision import datasets



# ============================================================
# SimCLR Dataset
#
# Creates two augmented views from the same image.
#
# Input:
#   Original CIFAR-10 image
#
# Output:
#   view_1, view_2
#
# Labels are ignored because this is self-supervised learning.
#
# ============================================================


class SimCLRDataset(Dataset):

    def __init__(
        self,
        root,
        transform,
        train=True,
        download=False,
    ):

        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
        )

        self.transform = transform



    def __len__(self):

        return len(self.dataset)



    def __getitem__(
        self,
        index,
    ):

        image, _ = self.dataset[index]


        # ----------------------------------------------------
        # Create two different augmented views
        # ----------------------------------------------------

        view_1 = self.transform(image)

        view_2 = self.transform(image)


        return view_1, view_2