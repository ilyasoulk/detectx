import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision import ops


class BipartiteLoss(nn.Module):
    def __init__(
        self, weight_dict, num_classes=92, num_queries=100, no_obj_weight=0.1
    ) -> None:
        super().__init__()
        self.weight_dict = weight_dict
        self.num_classes = num_classes
        self.no_obj_weight = no_obj_weight
        self.num_queries = num_queries
        empty_weight = torch.ones(num_classes)
        empty_weight[-1] = no_obj_weight
        self.register_buffer("empty_weight", empty_weight)

    @torch.no_grad()
    def hungarian_matching(self, yhat, y):
        cls_pred, bb_pred = yhat  # (B, N, num_cls), (B, N, 4)
        cls_gt, bb_gt = y  # (tuple of tensors), (tuple of tensors)
        B, N, num_cls = cls_pred.shape

        indices = []
        # Process each batch item separately
        for batch_idx in range(B):
            # Get predictions for this batch item
            out_prob = cls_pred[batch_idx].softmax(-1)  # (N, num_classes)
            out_bbox = bb_pred[batch_idx]  # (N, 4)

            # Get targets for this batch item
            tgt_ids = cls_gt[batch_idx]  # (M,) where M is number of objects
            tgt_bbox = bb_gt[batch_idx]  # (M, 4)

            # Skip if no targets
            if len(tgt_ids) == 0:
                indices.append(
                    (
                        torch.tensor([], dtype=torch.int64),
                        torch.tensor([], dtype=torch.int64),
                    )
                )
                continue

            # Compute costs
            cost_class = -out_prob[:, tgt_ids]  # (N, M)

            # Convert boxes for IoU calculation
            out_bbox_xyxy = ops.box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy")
            tgt_bbox_xyxy = ops.box_convert(tgt_bbox, in_fmt="cxcywh", out_fmt="xyxy")

            cost_giou = -ops.box_iou(out_bbox_xyxy, tgt_bbox_xyxy)  # (N, M)
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # (N, M)

            # Final cost matrix
            C = (
                self.weight_dict["ce"] * cost_class
                + self.weight_dict["bbox"] * cost_bbox
                + self.weight_dict["giou"] * cost_giou
            )

            # Do Hungarian matching
            pred_idx, tgt_idx = linear_sum_assignment(C.cpu())
            indices.append(
                (
                    torch.as_tensor(pred_idx, dtype=torch.int64),
                    torch.as_tensor(tgt_idx, dtype=torch.int64),
                )
            )

        return indices

    def forward(self, yhat, y):
        cls_pred, bb_pred = yhat
        cls_gt, bb_gt = y
        B, N, _ = cls_pred.shape
        indices = self.hungarian_matching(yhat, y)
        losses = {"loss_ce": 0, "loss_bbox": 0, "loss_giou": 0}

        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            # Initialize all predictions as "no object" class
            target_classes = torch.full(
                (N,), self.num_classes - 1, dtype=torch.int64, device=cls_pred.device
            )

            if len(pred_idx) > 0:  # Only update if we have matches
                target_classes[pred_idx] = cls_gt[batch_idx][tgt_idx]

            # Classification loss
            loss_ce = F.cross_entropy(
                cls_pred[batch_idx], target_classes, weight=self.empty_weight
            )
            losses["loss_ce"] += loss_ce

            if len(pred_idx) > 0:  # Only compute box losses if we have matches
                # Box loss
                src_boxes = bb_pred[batch_idx][pred_idx]
                target_boxes = bb_gt[batch_idx][tgt_idx]

                loss_bbox = F.l1_loss(src_boxes, target_boxes)
                losses["loss_bbox"] += loss_bbox

                # Convert to xyxy format
                src_boxes_xyxy = ops.box_convert(
                    src_boxes, in_fmt="cxcywh", out_fmt="xyxy"
                )
                target_boxes_xyxy = ops.box_convert(
                    target_boxes, in_fmt="cxcywh", out_fmt="xyxy"
                )

                loss_giou = (1 - ops.box_iou(src_boxes_xyxy, target_boxes_xyxy)).mean()
                losses["loss_giou"] += loss_giou

        # Normalize losses by batch size
        losses = {
            k: v * self.weight_dict[k.split("_")[1]] / B for k, v in losses.items()
        }
        losses["total"] = sum(losses.values())
        return losses
