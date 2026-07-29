import torch
import torch.nn as nn
import torch.nn.functional as F



# ============================================================
# NT-Xent Contrastive Loss
#
# SimCLR loss function.
#
# Positive pairs:
# Two augmented views of the same image.
#
# Negative pairs:
# All other images in the batch.
#
# ============================================================


class NTXentLoss(nn.Module):

    def __init__(
        self,
        temperature=0.07,
    ):

        super().__init__()

        self.temperature = temperature



    def forward(
        self,
        z1,
        z2,
    ):

        batch_size = z1.size(0)


        # ----------------------------------------------------
        # Normalize embeddings
        # ----------------------------------------------------

        z1 = F.normalize(
            z1,
            dim=1,
        )

        z2 = F.normalize(
            z2,
            dim=1,
        )


        # ----------------------------------------------------
        # Combine two views
        # Result:
        # (2 * batch_size, projection_dim)
        # ----------------------------------------------------

        representations = torch.cat(
            [
                z1,
                z2,
            ],
            dim=0,
        )


        # ----------------------------------------------------
        # Similarity matrix
        # ----------------------------------------------------

        similarity_matrix = torch.matmul(
            representations,
            representations.T,
        )


        # ----------------------------------------------------
        # Remove self similarity
        # ----------------------------------------------------

        mask = torch.eye(
            2 * batch_size,
            device=z1.device,
            dtype=torch.bool,
        )


        similarity_matrix = similarity_matrix.masked_fill(
            mask,
            -9e15,
        )


        # ----------------------------------------------------
        # Positive pair similarities
        #
        # First half:
        # z1 -> z2
        #
        # Second half:
        # z2 -> z1
        # ----------------------------------------------------

        positives = torch.cat(
            [
                torch.diag(
                    similarity_matrix,
                    batch_size,
                ),

                torch.diag(
                    similarity_matrix,
                    -batch_size,
                ),
            ],
            dim=0,
        )


        # ----------------------------------------------------
        # Temperature scaling
        # ----------------------------------------------------

        logits = similarity_matrix / self.temperature


        labels = torch.zeros(
            2 * batch_size,
            dtype=torch.long,
            device=z1.device,
        )


        # Move positive sample to first position
        logits = torch.cat(
            [
                positives.unsqueeze(1),
                logits,
            ],
            dim=1,
        )


        # Positive is always index 0
        loss = F.cross_entropy(
            logits,
            labels,
        )


        return loss