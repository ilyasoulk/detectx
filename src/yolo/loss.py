import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, num_classes=20, num_boxes=2, split_size=7, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.C = num_classes
        self.B = num_boxes
        self.S = split_size
        self.lambda_coord = lambda_coord # weight for the localization loss
        self.lambda_noobj = lambda_noobj # weight for the no-object confidence loss
        self.mse = nn.MSELoss(reduction="sum")
        
    def forward(self, preds, targets):
        preds = preds.reshape(-1, self.S, self.S, self.B * 5 + self.C)
        
        preds_bb1, preds_bb2 = preds[..., 21:25], preds[..., 26:30]
        target_bb = targets[..., 21:25]
        preds_score1, preds_score2 = preds[..., 20:21], preds[..., 25:26]
        target_score = targets[..., 20:21]
        
        iou_bb1 = intersection_over_union(preds_bb1, target_bb)
        iou_bb2 = intersection_over_union(preds_bb2, target_bb)
        ious = torch.cat([iou_bb1.unsqueeze(0), iou_bb2.unsqueeze(0)], dim=0)
        best_box = torch.argmax(ious, dim=0) # 0 if the first predicted box is the best, 1 otherwise
        box_present = targets[..., 20].unsqueeze(3) # Iobj_i -> is there an object
        
        # -------------------- #
        # BOX COORDINATES LOSS #
        # -------------------- #
        
        box_preds = box_present * (
            best_box * preds_bb2
            + (1 - best_box) * preds_bb1
        )
        box_targets = box_present * target_bb
        
        # sqrt for width and height
        box_preds[..., 2:4] = torch.sign(box_preds[..., 2:4]) * torch.sqrt(
            torch.abs(box_preds[..., 2:4] + 1e-6) # add 1e-6 because sqrt(0) is not derivable
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        # (B, S, S, 4) -> (B*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_preds, end_dim=-2), 
            torch.flatten(box_targets, end_dim=-2)
        )
        
        # ----------- #
        # OBJECT LOSS #
        # ----------- #
        
        # confidence score for the box with highest IoU
        pred_score = (
            best_box * preds_score2
            + (1 - best_box) * preds_score1
        )
        
        # (B*S*S, 1)
        object_loss = self.mse(
            torch.flatten(box_present * pred_score),
            torch.flatten(box_present * target_score)
        )
        
        # -------------- #
        # NO OBJECT LOSS #
        # -------------- #
        
        # (B, S, S, 1) -> (B, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - box_present) * preds_score1, start_dim=1),
            torch.flatten((1 - box_present) * target_score, start_dim=1)
        )
        no_object_loss += self.mse(
            torch.flatten((1 - box_present) * preds_score2, start_dim=1),
            torch.flatten((1 - box_present) * target_score, start_dim=1)
        )
        
        # ---------- #
        # CLASS LOSS #
        # ---------- #

        # (B, S, S, 20) -> (B*S*S, 20)
        class_loss = self.mse(
            torch.flatten(box_present * preds[..., :20], end_dim=-2),
            torch.flatten(box_present * targets[..., :20], end_dim=-2)
        )
        
        # ---------- #
        # FINAL LOSS #
        # ---------- #
        
        loss = (
            self.lambda_coord * box_loss # first and second rows of paper
            + object_loss # third row of paper
            + self.lambda_noobj * no_object_loss # fourth row of paper
            + class_loss # fifth row of paper
        )
        
        return loss
