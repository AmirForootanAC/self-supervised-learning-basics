import torch
import torch.nn as nn
from torchvision.models import resnet18



# ============================================================
# SimCLR Encoder
#
# Removes classification head from ResNet18.
#
# Output:
#   h = feature representation
#
# ============================================================


class SimCLREncoder(nn.Module):

    def __init__(self):

        super().__init__()


        backbone = resnet18(
            weights=None
        )


        # ----------------------------------------------------
        # CIFAR-10 adaptation
        #
        # Original ImageNet ResNet:
        #   7x7 conv + maxpool
        #
        # CIFAR-10:
        #   32x32 images
        #
        # ----------------------------------------------------

        backbone.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )


        backbone.maxpool = nn.Identity()



        # ----------------------------------------------------
        # Remove classification head
        # Keep convolutional feature extractor
        #
        # Output dimension:
        #   512
        #
        # ----------------------------------------------------

        self.encoder = nn.Sequential(
            *list(backbone.children())[:-1]
        )


        self.feature_dim = 512



    def forward(
        self,
        x,
    ):

        x = self.encoder(x)


        x = torch.flatten(
            x,
            start_dim=1,
        )


        return x





# ============================================================
# Projection Head
#
# Maps representation h -> projection z
#
# Used only during contrastive training.
#
# ============================================================


class ProjectionHead(nn.Module):

    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
        output_dim=128,
    ):

        super().__init__()


        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                output_dim,
            ),
        )



    def forward(
        self,
        x,
    ):

        return self.net(x)





# ============================================================
# Complete SimCLR Model
#
# Encoder + Projection Head
#
# Input:
#   Image
#
# Output:
#   Projection embedding z
#
# ============================================================


class SimCLR(nn.Module):

    def __init__(
        self,
        projection_dim=128,
    ):

        super().__init__()


        self.encoder = SimCLREncoder()


        self.projector = ProjectionHead(
            input_dim=self.encoder.feature_dim,
            output_dim=projection_dim,
        )



    def forward(
        self,
        x,
    ):

        h = self.encoder(x)


        z = self.projector(h)


        return z