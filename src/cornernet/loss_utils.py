from typing import Tuple

import torch


def contrastive(
    tl: torch.Tensor,
    br: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple(torch.Tensor, torch.Tensor):
    """
    Compute the push loss for the top left and bottom right corners of the bounding box.

    Args:
        tl (torch.Tensor): The top left corner of the bounding box.
        br (torch.Tensor): The bottom right corner of the bounding box.
        mask (torch.Tensor): A binary mask indicating valid entries for the computation (e.g., identifying which corner pairs correspond to objects in the batch).

    Returns:
        torch.Tensor: The push loss for the top left and bottom right corners of the bounding box.
    """
    # sum of valid entries for each bounding box in the mask
    valids = mask.sum(dim=1, keepdim=True).float()

    tl = tl.squeeze()
    br = br.squeeze()

    # mean
    m = (tl + br) / 2

    # pull loss, ensure normalization by the number of valid entries
    tl = torch.pow(tl - m, 2) / (valids + 1e-6)
    br = torch.pow(br - m, 2) / (valids + 1e-6)

    tl = tl[mask].sum()
    br = br[mask].sum()
    pull = tl + br

    # push loss
    # pairwise mask indicating which embeddings are from the same object
    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    valids = valids.unsqueeze(2)
    valids2 = (valids - 1) * valids

    # distance between top-left and bottom-right corners from the same class
    dist =
