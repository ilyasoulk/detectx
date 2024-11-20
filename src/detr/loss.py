import torch
import torch.nn as nn
import torch.nn.functional as F

class HungarianLoss(nn.Module):
    def __init__(self, lambda_l1=5, lambda_iou=2) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_iou = lambda_iou
        
    def forward(self, x, y):
        # X (classes, bboxes) ((B, N, num_cls), (B, N, 4))
        cls_pred, bb_pred = x
        cls_gt, bb_gt = y
        print(f"CLS : {cls_gt}")
        B, N, num_cls = cls_pred.shape
        
        # Classification Loss (Cross Entropy)
        cls_pred = F.log_softmax(cls_pred, dim=2)
        cls_loss = -torch.sum(cls_gt * cls_pred, dim=2)
        
        background_idx = num_cls - 1
        mask = (torch.argmax(cls_gt, dim=2) != background_idx).float()
        
        # L1 Loss
        l1_loss = self.lambda_l1 * (bb_gt - bb_pred).abs().sum(dim=2)
        
        # IoU Loss
        bb_pred_min = bb_pred[:, :, :2]
        bb_pred_max = bb_pred[:, :, 2:]
        bb_gt_min = bb_gt[:, :, :2]
        bb_gt_max = bb_gt[:, :, 2:]
        
        # Calculate intersection
        bb_int_min = torch.max(bb_pred_min, bb_gt_min)
        bb_int_max = torch.min(bb_pred_max, bb_gt_max)
        bb_int = torch.clamp(bb_int_max - bb_int_min, min=0)
        intersection_area = bb_int[..., 0] * bb_int[..., 1]
        
        # Calculate areas
        bb_pred_area = (bb_pred[:, :, 2] - bb_pred[:, :, 0]) * (bb_pred[:, :, 3] - bb_pred[:, :, 1])
        bb_gt_area = (bb_gt[:, :, 2] - bb_gt[:, :, 0]) * (bb_gt[:, :, 3] - bb_gt[:, :, 1])
        
        # Calculate IoU
        union = bb_pred_area + bb_gt_area - intersection_area
        iou = intersection_area / (union + 1e-6)
        iou_loss = self.lambda_iou * (1 - iou)
        
        # combine losses with mask
        bb_loss = mask * (iou_loss + l1_loss)
        total_loss = cls_loss + bb_loss
        
        loss = total_loss.sum() / (N * B)
        
        return loss

if __name__ == "__main__":
    # Test the implementation
    B, N, num_cls = 2, 100, 21  # 20 classes + 1 background
    
    cls_pred = torch.randn(B, N, num_cls)
    bbox_pred = torch.randn(B, N, 4)
    
    cls_gt = torch.zeros(B, N, num_cls)
    cls_gt[:, :, :-1].random_(0, 2)
    cls_gt[:, :, -1] = 1 - cls_gt[:, :, :-1].sum(dim=2)
    
    bbox_gt = torch.randn(B, N, 4)
    
    # Test the loss
    criterion = HungarianLoss(lambda_l1=1.0, lambda_iou=1.0)
    loss = criterion((cls_pred, bbox_pred), (cls_gt, bbox_gt))
    print(f"Loss shape: {loss.shape}")
    print(f"Loss value: {loss.item()}")
