import torch
from dataset import (
    MyCocoDetection,
    plot_im_with_boxes,
    revert_normalization,
    CLASSES,
    collate_fn,
)

# from dataset_voc import (
#     MyVOCDetection,
#     plot_im_with_boxes,
#     revert_normalization,
#     CLASSES,
#     collate_fn,
# )
import matplotlib.pyplot as plt
from torchvision import ops
from engine import build
from torch.utils.data import DataLoader
import torch.nn.functional as F
from metrics import calculate_metrics


def visualize_predictions(model, dataloader, device, criterion, num_images=5):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            if idx >= num_images:
                break

            # Move data to device
            image = images[0]  # Take first image from batch
            images = images.to(device)
            classes_target, bbox_target = labels
            classes_device = tuple(c.to(device) for c in classes_target)
            bbox_device = tuple(b.to(device) for b in bbox_target)
            labels_device = (classes_device, bbox_device)

            # Get model predictions
            outputs = model(images)
            classes_pred, boxes_pred = outputs

            # Move predictions to CPU before processing
            classes_pred = classes_pred.cpu()
            boxes_pred = boxes_pred.cpu()

            # Calculate losses - move labels to CPU first
            labels_cpu = (
                tuple(c.cpu() for c in classes_device),
                tuple(b.cpu() for b in bbox_device),
            )
            outputs_cpu = (classes_pred, boxes_pred)
            loss_dict = criterion(outputs_cpu, labels_cpu)

            for k, v in loss_dict.items():
                print(f"For loss {k} the value is : {v}")

            # Process predictions for first image
            probs = F.softmax(classes_pred[0], dim=-1)
            boxes = boxes_pred[0]

            # Convert boxes from cxcywh to xyxy format
            boxes_xyxy = ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

            # Get prediction scores and classes
            max_probs, pred_classes = probs.max(dim=1)

            # Filter predictions
            conf_threshold = 0.6
            keep = (max_probs > conf_threshold) & (pred_classes != (len(CLASSES) - 1))
            filtered_boxes = boxes_xyxy[keep]
            filtered_probs = probs[keep]

            print(f"Targets : {labels}")
            print(f"Predicted classes : {pred_classes[keep]}")

            # Image processing
            img_show = revert_normalization(image)
            img_show = torch.clamp(img_show, 0, 1)

            # Visualization
            plt.figure(figsize=(12, 8))
            plot_im_with_boxes(
                img_show.permute(1, 2, 0).cpu(),
                filtered_boxes * img_show.shape[1],  # Scale boxes to image size
                filtered_probs,
            )
            plt.title(f"Image {idx}")
            plt.show()


weight_dict = {"ce": 5, "bbox": 2, "giou": 1}
device = "mps"
model, criterion, _, _ = build(
    weight_dict,
    backbone="resnet18",
    hidden_dim=128,
    num_heads=4,
    num_encoder=2,
    num_decoder=2,
    num_cls=92,
)

batch_size = 1

train_ds = MyCocoDetection(
    "tiny_coco_dataset/tiny_coco/train2017/",
    "tiny_coco_dataset/tiny_coco/annotations/instances_train2017.json",
)

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

# train_ds = MyVOCDetection(
#     root="./data_voc", year="2012", image_set="train", download=True
# )
#
# train_loader = DataLoader(
#     train_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
# )
model_path = "models/tiny_model.pth"
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
visualize_predictions(model, train_loader, device, criterion)
