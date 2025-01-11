# loss used in corner net is a combination of:
# 1. a variant of focal loss for heatmap prediction
# 2. smooth l1 loss for box prediction and counter the loss of precision made when remapping the box from the heatmap to the original image
# 3. "pull" loss to train the network to group the corner of the same object together
# 4. "push" loss to train the network to separate the corner of different objects
# third and fourth losses are used to detect similar objects (distance between top-left and bottom-right corners from the same class and
# different objects should be close)

import torch
import torch.nn as nn

class CornerLoss(nn.Module):
    def __init__(self, theta: float = 1.0, alpha: float = 2.0, beta: float = 4.0):
        super(CornerLoss, self).__init__()

        self.theta = theta
        self.alpha = alpha
        self.beta = beta

        self.
