import torch
import torch.nn as nn
import torch.nn.functional as F

class YoloLoss(nn.Module):
    def __init__(self, num_classes=20, num_anchors=2, img_dim=448, cell_dim=7, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.img_dim = img_dim
        self.cell_dim = cell_dim
        self.lambda_coord = lambda_coord # weight for the localization loss
        self.lambda_noobj = lambda_noobj # weight for the no-object confidence loss
        
    def forward(self, outputs, targets):
        pred_boxes = outputs[..., :4]
        pred_obj_conf = outputs[..., 4:5]
        pred_cls = outputs[..., 5:]
        
        target_boxes = targets[..., :4]
        target_obj_conf = targets[..., 4:5]
        target_cls = targets[..., 5:]
        
        # Localization loss
        coord_mask = target_obj_conf # 1 if object exists in cell, else 0
        coord_loss = self.lambda_coord * torch.sum(coord_mask * F.mse_loss(pred_boxes, target_boxes, reduction="none"))

        # Confidence loss
        obj_mask = target_obj_conf
        noobj_mask = 1 - obj_mask
        conf_loss_obj = torch.sum(obj_mask * F.mse_loss(pred_obj_conf, target_obj_conf, reduction="none"))
        conf_loss_noobj = torch.sum(noobj_mask * F.mse_loss(pred_obj_conf, target_obj_conf, reduction="none"))
        conf_loss = conf_loss_obj + conf_loss_noobj * self.lambda_noobj
        
        class_loss = torch.sum(obj_mask * F.cross_entropy(pred_cls, target_cls, reduction="none"))
        
        total_loss = coord_loss + conf_loss + class_loss
        
        return total_loss

random_outputs = torch.rand(16, 7, 7, 25)
random_targets = torch.rand(16, 7, 7, 25)

criterion = YoloLoss()
loss = criterion(random_outputs, random_targets)

print(loss)
