from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import torchvision.ops as ops


def calculate_metrics(
    pred_boxes, pred_labels, pred_scores, target_boxes, target_labels, iou_threshold=0.5
):
    """Calculate mAP using torchmetrics"""
    metric = MeanAveragePrecision(box_format="xyxy")

    # Filter predictions
    keep_mask = pred_labels != 22  # TODO : make this an arugment
    confidence_mask = pred_scores > 0.5
    final_mask = keep_mask & confidence_mask

    pred_boxes = pred_boxes[final_mask]
    pred_labels = pred_labels[final_mask]
    pred_scores = pred_scores[final_mask]

    # Convert both to xyxy
    target_boxes = ops.box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy")

    # Calculate and print IoUs between first few predictions and targets
    ious = ops.box_iou(pred_boxes[:3], target_boxes[:3])
    print(f"\nIoUs for first 3 boxes:\n{ious}")

    preds = [
        {
            "boxes": pred_boxes,
            "scores": pred_scores,
            "labels": pred_labels,
        }
    ]

    targets = [
        {
            "boxes": target_boxes,
            "labels": target_labels,
        }
    ]

    metric.update(preds, targets)
    results = metric.compute()

    formatted_results = {}
    for k, v in results.items():
        if v.numel() == 1:
            formatted_results[k] = v.item()
        else:
            v_float = v.float() if v.dtype in [torch.int32, torch.int64] else v
            formatted_results[k] = v_float.mean().item()

    print("MAP Results:", formatted_results)
    return results["map"].item()
