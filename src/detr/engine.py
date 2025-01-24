import torch
from torch.optim import AdamW
from model import DeTr
from loss import BipartiteLoss
from tqdm import tqdm
from metrics import calculate_metrics
from torchvision import ops


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, max_norm):
    model.train()
    criterion.train()
    metric_logger = {}
    metric_logger["loss_ce"] = 0.0
    metric_logger["loss_bbox"] = 0.0
    metric_logger["loss_giou"] = 0.0
    metric_logger["total"] = 0.0
    elements = 0

    # Track predictions and targets for mAP calculation
    all_predictions = []
    all_targets = []

    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")

    for images, labels in pbar:
        images = images.to(device)
        classes = tuple(label.to(device) for label in labels[0])
        bbox = tuple(label.to(device) for label in labels[1])
        labels = (classes, bbox)

        outputs = model(images)
        loss_dict = criterion(outputs, labels)
        weight_dict = criterion.weight_dict

        # Store predictions and targets for mAP calculation
        pred_logits, pred_boxes = outputs
        pred_probs = pred_logits.softmax(-1)
        pred_scores, pred_labels = pred_probs.max(-1)

        # Reshape predictions to be [batch_size * num_queries, ...]
        pred_boxes = ops.box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
        all_predictions.append(
            {
                "boxes": pred_boxes.view(
                    -1, 4
                ),  # Flatten to [batch_size * num_queries, 4]
                "scores": pred_scores.view(-1),  # Flatten to [batch_size * num_queries]
                "labels": pred_labels.view(-1),  # Flatten to [batch_size * num_queries]
            }
        )

        # Store batch targets
        all_targets.append(
            {
                "boxes": bbox[0],  # Already in correct shape [num_targets, 4]
                "labels": classes[0],  # Already in correct shape [num_targets]
            }
        )

        losses = loss_dict["total"]

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # Accumulate all losses without batch normalization
        for k, v in loss_dict.items():
            metric_logger[k] += v.item()

        elements += 1

        # For progress bar, show current running averages
        current_metrics = {f"{k}": v / elements for k, v in metric_logger.items()}
        pbar.set_postfix(current_metrics)

    # Calculate mAP with accumulated batches
    mAP = calculate_metrics(
        torch.cat(
            [p["boxes"] for p in all_predictions]
        ),  # Shape: [total_predictions, 4]
        torch.cat([p["labels"] for p in all_predictions]),  # Shape: [total_predictions]
        torch.cat([p["scores"] for p in all_predictions]),  # Shape: [total_predictions]
        torch.cat([t["boxes"] for t in all_targets]),  # Shape: [total_targets, 4]
        torch.cat([t["labels"] for t in all_targets]),  # Shape: [total_targets]
    )

    # Final metrics are all normalized by total number of elements
    avg_loss = metric_logger["total"] / elements
    metrics = {k: v / elements for k, v in metric_logger.items()}
    metrics["mAP"] = mAP

    return avg_loss, metrics


def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()
    running_loss = 0.0
    elements = 0
    metric_logger = {}

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating")

        for images, labels in pbar:
            images = images.to(device)
            classes = tuple(label.to(device) for label in labels[0])
            bbox = tuple(label.to(device) for label in labels[1])
            labels = (classes, bbox)

            outputs = model(images)
            loss_dict = criterion(outputs, labels)
            weight_dict = criterion.weight_dict

            # Store predictions and targets for mAP calculation
            pred_logits, pred_boxes = outputs
            pred_probs = pred_logits.softmax(-1)
            pred_scores, pred_labels = pred_probs.max(-1)

            pred_boxes = ops.box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")

            # Reshape predictions to match expected format
            pred_boxes = pred_boxes.view(-1, 4)  # Reshape to (N, 4)
            pred_scores = pred_scores.view(-1)  # Reshape to (N,)
            pred_labels = pred_labels.view(-1)  # Reshape to (N,)

            # Store batch predictions
            all_predictions.append(
                {
                    "boxes": pred_boxes.detach(),
                    "scores": pred_scores.detach(),
                    "labels": pred_labels.detach(),
                }
            )

            # Store batch targets
            all_targets.append(
                {
                    "boxes": bbox[0],  # Assuming bbox is a tuple with one tensor
                    "labels": classes[0],  # Assuming classes is a tuple with one tensor
                }
            )

            losses = loss_dict["total"]

            for k, v in loss_dict.items():
                if k in weight_dict:
                    metric_logger[k] = metric_logger.get(k, 0.0) + v.item()

            running_loss += losses.item()
            elements += 1

            pbar.set_postfix(
                {
                    "val_loss": running_loss / elements,
                    **{f"val_{k}": v / elements for k, v in metric_logger.items()},
                }
            )

    # Calculate mAP with accumulated batches
    mAP = calculate_metrics(
        torch.cat([p["boxes"] for p in all_predictions]),
        torch.cat([p["labels"] for p in all_predictions]),
        torch.cat([p["scores"] for p in all_predictions]),
        torch.cat([t["boxes"] for t in all_targets]),
        torch.cat([t["labels"] for t in all_targets]),
    )

    avg_loss = running_loss / elements
    metrics = {k: v / elements for k, v in metric_logger.items()}
    metrics["mAP"] = mAP

    return avg_loss, metrics


def build(
    weight_dict,
    backbone="resnet50",
    hidden_dim=256,
    input_shape=(3, 512, 512),
    fc_dim=2048,
    num_heads=8,
    activ_fn="relu",
    num_encoder=6,
    num_decoder=6,
    num_obj=100,
    num_cls=92,
    learning_rate=1e-4,
    weight_decay=1e-4,
):
    model = DeTr(
        backbone=backbone,
        hidden_dim=hidden_dim,
        input_shape=input_shape,
        fc_dim=fc_dim,
        num_heads=num_heads,
        activ_fn=activ_fn,
        num_encoder=num_encoder,
        num_decoder=num_decoder,
        num_obj=num_obj,
        num_cls=num_cls,
    )

    criterion = BipartiteLoss(weight_dict, num_classes=num_cls)

    params = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n],
            "lr": learning_rate * 0.1,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": learning_rate,
        },
    ]

    optimizer = AdamW(params, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    return model, criterion, optimizer, scheduler
