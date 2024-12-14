import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)

    x_min = torch.maximum(boxes1[:, None, 0], boxes2[:, 0])  # (N,M)
    y_min = torch.maximum(boxes1[:, None, 1], boxes2[:, 1])  # (N,M)
    x_max = torch.minimum(boxes1[:, None, 2], boxes2[:, 2])  # (N,M)
    y_max = torch.minimum(boxes1[:, None, 3], boxes2[:, 3])  # (N,M)

    intersection = torch.clamp(x_max - x_min, min=0) * torch.clamp(
        y_max - y_min, min=0
    )  # (N,M)

    union = area1[:, None] + area2 - intersection  # (N,M)

    iou = intersection / (union + 1e-6)

    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def box_cxcywh_to_xyxy(x):
    """Convert boxes from (x, y, w, h) to (xmin, ymin, xmax, ymax)"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """Convert boxes from (xmin, ymin, xmax, ymax) to (x, y, w, h)"""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


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

        out_prob = cls_pred.flatten(0, 1).softmax(
            -1
        )  # [batch_size * num_queries, num_classes]
        out_bbox = bb_pred.flatten(0, 1)  # [batch_size * num_queries, 4]

        tgt_ids = torch.cat(cls_gt)
        tgt_bbox = torch.cat(bb_gt)
        # Classification Loss (Cross Entropy)
        cost_class = -out_prob[:, tgt_ids]

        # L2 Loss
        l1_loss = torch.cdist(out_bbox, tgt_bbox, p=1)

        # IoU Loss
        iou = -box_iou(out_bbox, tgt_bbox)[0]
        # giou = -generalized_box_iou(out_bbox, tgt_bbox)
        # giou = -generalized_box_iou(
        #     box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        # )

        C = (
            self.lambda_iou * iou
            + self.lambda_l1 * l1_loss
            + self.lambda_cls * cost_class
        )
        C = C.view(B, N, -1).cpu()
        sizes = [len(gt) for gt in cls_gt]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        indices = [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
        return indices


class HungarianLoss(nn.Module):
    def __init__(self, matcher, num_classes, no_obj_weight=0.1) -> None:
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.no_obj_weight = no_obj_weight

        empty_weight = torch.ones(num_classes)
        empty_weight[-1] = no_obj_weight

        self.register_buffer("empty_weight", empty_weight)

    def forward(self, yhat, y):
        # X (classes, bboxes) ((B, N, num_cls), (B, N, 4))
        cls_pred, bb_pred = yhat
        cls_gt, bb_gt = y
        B, N, _ = cls_pred.shape
        indices = self.matcher(yhat, y)

        # Compute final losses using matched pairs
        losses = {"loss_ce": 0.0, "loss_bbox": 0.0, "loss_iou": 0.0}

        # Classification loss for matched pairs
        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            # Create target tensor with background class (num_cls) as default
            target_classes = torch.full(
                (N,), self.num_classes - 1, dtype=torch.int64, device=cls_pred.device
            )
            target_classes[pred_idx] = cls_gt[batch_idx][tgt_idx]
            loss_ce = F.cross_entropy(
                cls_pred[batch_idx], target_classes, weight=self.empty_weight
            )
            losses["loss_ce"] += loss_ce

            # Box loss for matched pairs
            src_boxes = bb_pred[batch_idx][pred_idx]
            target_boxes = bb_gt[batch_idx][tgt_idx]
            l1_loss = F.l1_loss(src_boxes, target_boxes)
            losses["loss_bbox"] += l1_loss

            # IoU loss for matched pairs
            iou = box_iou(src_boxes, target_boxes)[0]
            iou = (1 - iou.diagonal()).mean()

            # giou = generalized_box_iou(
            #     box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            # )
            # giou = (1 - giou.diagonal()).mean()
            losses["loss_iou"] += iou

        # Normalize losses by batch size
        losses = {k: v / B for k, v in losses.items()}

        # Compute total loss with weights
        total_loss = losses["loss_ce"] + losses["loss_bbox"] + losses["loss_iou"]

        losses["total_loss"] = total_loss
        return losses


def visualize_boxes(pred_boxes, gt_boxes, matched_indices=None, title="Bounding Boxes"):
    """Helper function to visualize boxes and their matches"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot predicted boxes in blue
    for i, box in enumerate(pred_boxes):
        rect = Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            color="blue",
            label="Predicted" if i == 0 else "",
        )
        ax.add_patch(rect)
        ax.text(box[0], box[1], f"P{i}", color="blue")

    # Plot ground truth boxes in red
    for i, box in enumerate(gt_boxes):
        rect = Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            color="red",
            label="Ground Truth" if i == 0 else "",
        )
        ax.add_patch(rect)
        ax.text(box[0], box[1], f"G{i}", color="red")

    # Draw matching lines if provided
    if matched_indices is not None:
        pred_idx, gt_idx = matched_indices
        for p_idx, g_idx in zip(pred_idx, gt_idx):
            p_box = pred_boxes[p_idx]
            g_box = gt_boxes[g_idx]
            p_center = [(p_box[0] + p_box[2]) / 2, (p_box[1] + p_box[3]) / 2]
            g_center = [(g_box[0] + g_box[2]) / 2, (g_box[1] + g_box[3]) / 2]
            ax.plot(
                [p_center[0], g_center[0]], [p_center[1], g_center[1]], "g--", alpha=0.5
            )

    ax.legend()
    ax.set_title(title)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    plt.show()


# Test Case 1: Perfect alignment
def test_perfect_alignment(matcher, loss_fn):
    # Predicted boxes and classes
    bb_pred = torch.tensor(
        [
            [[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.5, 0.5], [0.4, 0.4, 0.8, 0.8]]
        ],  # Box 1  # Box 2
        dtype=torch.float32,
    )

    cls_pred = torch.zeros((1, 3, 3))  # 3 classes (including background)
    cls_pred[0, 0, 0] = 5.0  # High confidence for class 0 in first box
    cls_pred[0, 1, 0] = 5.0
    cls_pred[0, 2, 1] = 5.0
    # cls_pred[0, 1, 2] = 5.0
    # cls_pred[0, 2, 1] = 5.0  # High confidence for class 1 in second box

    # Ground truth boxes and classes
    bb_gt = [
        torch.tensor(
            [
                [0.1, 0.1, 0.3, 0.3],
                [0.25, 0.28, 0.43, 0.49],
                [0, 0, 0, 0],
            ],  # Box 1  # Box 2
            dtype=torch.float32,
        )
    ]

    cls_gt = [torch.tensor([0, 0, 2])]  # Class 0 for first box, class 1 for second box

    # Get predictions and compute loss
    yhat = (cls_pred, bb_pred)
    y = (cls_gt, bb_gt)

    indices = matcher(yhat, y)
    loss = loss_fn(yhat, y)

    print("Test Case 1: Perfect Alignment")
    print(f"Matched indices: {indices}")
    print(
        f"Loss: {loss}"
    )  # Loss should be around 0.013 since we match 0 with 0 and 1 with 2. l1_loss = 0, GIoU = 0, nll = log(e**5 + 2) - 5 = 0.013

    visualize_boxes(bb_pred[0], bb_gt[0], indices[0], "Test Case 1: Perfect Alignment")
    return loss


# Test Case 2: Slight misalignment
def test_slight_misalignment(matcher, loss_fn):
    # Predicted boxes and classes
    bb_pred = torch.tensor(
        [
            [
                [0.65, 0.65, 0.85, 0.85],  # Box 2 (slightly offset)
                [0.15, 0.15, 0.35, 0.35],
            ]  # Box 1 (slightly offset)
        ],
        dtype=torch.float32,
    )

    cls_pred = torch.zeros((1, 2, 3))
    cls_pred[0, 0, 1] = 15.0
    cls_pred[0, 1, 0] = 10.0

    # Ground truth boxes and classes
    bb_gt = [
        torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.8, 0.8]], dtype=torch.float32)
    ]

    cls_gt = [torch.tensor([0, 1])]

    # Get predictions and compute loss
    yhat = (cls_pred, bb_pred)
    y = (cls_gt, bb_gt)

    indices = matcher(yhat, y)
    loss = loss_fn(yhat, y)

    print("\nTest Case 2: Slight Misalignment")
    print(f"Matched indices: {indices}")
    print(f"Loss: {loss}")

    visualize_boxes(
        bb_pred[0], bb_gt[0], indices[0], "Test Case 2: Slight Misalignment"
    )
    return loss


# Run the tests
if __name__ == "__main__":
    matcher = HungarianMatcher()
    loss_fn = HungarianLoss(matcher, num_classes=3)

    loss1 = test_perfect_alignment(matcher, loss_fn)
    loss2 = test_slight_misalignment(matcher, loss_fn)

    print("\nSummary:")
    print(f"Perfect alignment loss: {loss1}")
    print(f"Slight misalignment loss: {loss2}")
    print(f"Loss difference: {loss2 - loss1}")
