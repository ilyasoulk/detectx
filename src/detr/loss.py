import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom
    
    wh = (rb - lt).clamp(min=0)  # intersection
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / (union + 1e-6)
    return iou

class HungarianMatcher(nn.Module):
    def __init__(self, lambda_l1=5.0, lambda_iou=2.0, lambda_cls=1.0):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_iou = lambda_iou
        self.lambda_cls = lambda_cls

    @torch.no_grad()
    def forward(self, yhat, y):
        # X (classes, bboxes) ((B, N, num_cls), (B, N, 4))
        cls_pred, bb_pred = yhat
        cls_gt, bb_gt = y
        B, N, num_cls = cls_pred.shape

        out_prob = cls_pred.flatten(0, 1).softmax(-1) # [batch_size * num_queries, num_classes]
        out_bbox = bb_pred.flatten(0, 1) # [batch_size * num_queries, 4]

        tgt_ids = torch.cat(cls_gt)
        tgt_bbox = torch.cat(bb_gt)
        # Classification Loss (Cross Entropy)
        cost_class = -out_prob[:, tgt_ids]

        # L1 Loss
        l1_loss = torch.cdist(out_bbox, tgt_bbox, p=1)

        # IoU Loss
        iou = box_iou(out_bbox, tgt_bbox)

        C = self.lambda_iou * iou + self.lambda_l1 * l1_loss + self.lambda_cls * cost_class
        C = C.view(B, N, -1).cpu()
        sizes = [len(gt) for gt in cls_gt]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), 
            torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        return indices


class HungarianLoss(nn.Module):
    def __init__(self, matcher) -> None:
        super().__init__()
        self.matcher = matcher

    def forward(self, yhat, y):
        # X (classes, bboxes) ((B, N, num_cls), (B, N, 4))
        cls_pred, bb_pred = yhat
        cls_gt, bb_gt = y
        B, N, num_cls = cls_pred.shape
        indices = self.matcher(yhat, y)

        # Compute final losses using matched pairs
        total_loss = 0
        
        # Classification loss for matched pairs
        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            # Create target tensor with background class (num_cls) as default
            target_classes = torch.full((N,), num_cls-1, 
                                     dtype=torch.int64, 
                                     device=cls_pred.device)
            target_classes[pred_idx] = cls_gt[batch_idx][tgt_idx]
            total_loss += F.cross_entropy(cls_pred[batch_idx], target_classes)
            
            # Box loss for matched pairs
            src_boxes = bb_pred[batch_idx][pred_idx]
            target_boxes = bb_gt[batch_idx][tgt_idx]
            total_loss += F.l1_loss(src_boxes, target_boxes)
            
            # IoU loss for matched pairs
            iou = box_iou(src_boxes, target_boxes)
            total_loss += (1 - iou.diagonal()).mean()
        
        return total_loss / B


if __name__ == "__main__":
    # Create a simple test case with 1 batch, 2 boxes
    B, N = 1, 2
    num_cls = 21  # 20 classes + 1 background

    # Create predictions (intentionally different from ground truth)
    cls_pred = torch.zeros(B, N, num_cls)
    cls_pred[0, 0, 0] = 5.0  # High confidence for class 0
    cls_pred[0, 1, 1] = 5.0  # High confidence for class 1

    # Predicted boxes: [x_min, y_min, x_max, y_max]
    bbox_pred = torch.tensor(
        [[[0.4, 0.4, 0.6, 0.6], [0.5, 0.5, 0.8, 0.8]]], dtype=torch.float32
    )

    # Create ground truth
    cls_gt = torch.zeros(B, N, num_cls)
    cls_gt[0, 0, 0] = 1.0  # First box is class 0
    cls_gt[0, 1, 1] = 1.0  # Second box is class 1

    # Ground truth boxes: [x_min, y_min, x_max, y_max]
    bbox_gt = torch.tensor(
        [[[0.15, 0.15, 0.35, 0.35], [0.6, 0.6, 0.9, 0.9]]], dtype=torch.float32
    )

    # Test the loss
    criterion = HungarianLoss(lambda_l1=1.0, lambda_iou=1.0)
    loss = criterion((cls_pred, bbox_pred), (cls_gt, bbox_gt))

    # Calculate IoU manually for verification
    def calculate_iou(box1, box2):
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate areas
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Calculate IoU
        union = area1 + area2 - intersection
        return intersection / (union + 1e-6)

    # Calculate IoU for both boxes
    iou1 = calculate_iou(bbox_pred[0, 0], bbox_gt[0, 0])
    iou2 = calculate_iou(bbox_pred[0, 1], bbox_gt[0, 1])

    print(f"Loss value: {loss.item()}")
    print(f"IoU for box 1: {iou1.item():.4f}")
    print(f"IoU for box 2: {iou2.item():.4f}")

    # Visualize the boxes
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Plot predicted boxes in red
    for i in range(N):
        box = bbox_pred[0, i]
        rect = Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            color="red",
            label="Predicted" if i == 0 else None,
        )
        ax.add_patch(rect)

    # Plot ground truth boxes in green
    for i in range(N):
        box = bbox_gt[0, i]
        rect = Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            color="green",
            label="Ground Truth" if i == 0 else None,
        )
        ax.add_patch(rect)

    ax.legend()
    plt.show()
