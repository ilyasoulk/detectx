# loss used in corner net is a combination of:
# 1. a variant of focal loss for heatmap prediction
# 2. smooth l1 loss for box prediction and counter the loss of precision made when remapping the box from the heatmap to the original image
# 3. "pull" loss to train the network to group the corner of the same object together
# 4. "push" loss to train the network to separate the corner of different objects
# third and fourth losses are used to detect similar objects (distance between top-left and bottom-right corners from the same class and
# different objects should be close)

import torch
import torch.nn as nn

def regr_loss(output, target, mask):
    num = mask.float().sum() * 2 # 2 for x and y

    mask = mask.unsqueeze(2).expand_as(target)
    output = output[mask == 1] # only consider the valid entries -> part of the bounding box
    target = target[mask == 1]

    res = nn.functional.smooth_l1_loss(output, target, size_average=False)
    res = res / (num + 1e-6) # ensure normalization by the number of valid entries
    return res

def focal_loss(output, target):
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    loss = 0
    for idx, pred in enumerate(output):
        neg_weights = torch.pow(1 - target[idx], 4)
        pos_pred = pred[pos_inds[idx] == 1.]
        neg_pred = pred[neg_inds[idx] == 1.]

        # compute the loss for positive and negative examples
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds[idx].sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss -= neg_loss # maybe += instead of -=
        else:
            loss -= (pos_loss + neg_loss) / num_pos # normalize by the number of positive examples

    return loss

def pp_loss(tag0, tag1, mask):
    """
    compute the pull and push loss for the top left and bottom right corners of the bounding box.
    push loss is the distance between top-left and bottom-right corners from the same class
    pull loss is the distance between top-left and bottom-right corners from different classes
    we learn to distinguish between the corners of the same object and the corners of different objects
    """
    num  = mask.sum(dim=1, keepdim=True).unsqueeze(1).expand_as(tag0)

    mask = mask.unsqueeze(2)
    tag_mean = (tag0 + tag1) / 2
    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = (tag0*mask).sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = (tag1*mask).sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    eye = torch.eye(100,100).unsqueeze(2).expand_as(mask).to(mask.device)
    mask = mask - eye
    mask = mask.eq(2)
    num  = num.unsqueeze(2).expand_as(mask)

    num2 = (num - 1) * num

    m=1

    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = m - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    # dist = dist - m / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()

    return pull, push


class CornerLoss(nn.Module):
    def __init__(self, theta: float = 1.0, alpha: float = 2.0, beta: float = 4.0):
        super(CornerLoss, self).__init__()

        self.theta = theta
        self.alpha = alpha
        self.beta = beta

        self.focal_loss = focal_loss
        self.regr_loss = regr_loss
        self.ae = pp_loss

    def forward(self, outputs, targets):
        """
        Args:
            outputs: list of torch.Tensor, output of the network
            targets: list of torch.Tensor, ground truth targets
        """
        tl_heats, br_heats, tl_tags, br_tags, tl_regrs, br_regrs, masks = targets

        tl_heats = tl_heats[-1]
        br_heats = br_heats[-1]
        tl_tags = tl_tags[-1]
        br_tags = br_tags[-1]
        tl_regrs = tl_regrs[-1]
        br_regrs = br_regrs[-1]
        mask = masks[-1]

        tl_heats = torch.clamp(tl_heats.sigmoid_(), min=1e-4, max=1 - 1e-4)
        br_heats = torch.clamp(br_heats.sigmoid_(), min=1e-4, max=1 - 1e-4)

        focal_loss = self.focal_loss(tl_heats, tl_tags) + self.focal_loss(br_heats, br_tags)
        regr_loss = self.regr_loss(tl_regrs, tl_tags, mask) + self.regr_loss(br_regrs, br_tags, mask)
        pull_loss, push_loss = self.ae(tl_tags, br_tags, mask)

        total_loss = focal_loss + self.alpha * regr_loss + self.beta * (pull_loss + push_loss)
        return total_loss
