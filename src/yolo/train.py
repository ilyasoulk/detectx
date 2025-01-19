from dataset import PascalVOCDataset
from model import YOLOv1
from loss import YoloLoss
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# import cv2
import wandb
from utils import (
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)

torch.manual_seed(seed=42)

LOAD_MODEL = False
LOAD_PATH = "src/yolo/checkpoints/overfit.pth.tar"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 2e-4
BATCH_SIZE = 16
WEIGHT_DECAY = 0
NUM_EPOCHS = 1000

# wandb_track = False

# if wandb_track:
#     wandb.init(
#         project="object-detection",
        
#         config={
#             "learning_rate": 0.001,
#             "architecture": "YOLOv1",
#             "dataset": "PASCAL VOC 2012",
#             "epochs": 2,
#         },
#     )

# Define basic transformations, resize to 448x448
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
dataset = PascalVOCDataset(
    root='~/data/datasets',  # where to save the dataset
    year='2012',    # can use '2007' or '2012'
    image_set='train',  # can be 'train', 'val', or 'trainval'
    transform=transform
)

# Split the dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
    
# Define the model
model = YOLOv1(input_shape=(3,448,448), split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
criterion = YoloLoss(num_classes=20, num_boxes=2, split_size=7, lambda_coord=5, lambda_noobj=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

if LOAD_MODEL:
    load_checkpoint(torch.load(LOAD_PATH), model, optimizer)

def train(train_loader, model, optimizer, criterion):
    mean_loss = []
    model.train()
    for images, targets in train_loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)
        mean_loss.append(loss.item())
        loss.backward()
        optimizer.step()
                
    print(f"Epoch {epoch + 1}, Loss: {sum(mean_loss)/len(mean_loss)}")
    # if wandb_track:
    #     wandb.log({"train/epoch_loss": loss.item()})
    
def validate(val_loader, model, criterion):
    mean_loss = []
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            output = model(images)
            loss = criterion(output, targets)
            mean_loss.append(loss.item())
        print(f"Validation Loss: {sum(mean_loss)/len(mean_loss)}")
        # if wandb_track:
        #     wandb.log({"val/loss": loss.item()})
    
print("Starting training...")
best_map = None

for epoch in range(NUM_EPOCHS):
    pred_boxes, target_boxes = get_bboxes(
        train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
    )
    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    )
    print(f"Train mAP: {mean_avg_prec}")
    
    if (not best_map and mean_avg_prec > 0.05) or (best_map and mean_avg_prec > best_map):
        checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=LOAD_PATH)
        best_map = mean_avg_prec
        
    
    train(train_loader=train_loader, model=model, optimizer=optimizer, criterion=criterion)
    validate(val_loader=val_loader, model=model, criterion=criterion)

if not best_map:
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, filename=LOAD_PATH)
    
    
    

# if wandb_track:
#     wandb.finish()

# show a random image from the dataset, with bounding boxes
# idx = np.random.randint(len(dataset))
# img, target = dataset[idx]
# img = img.permute(1, 2, 0).numpy()
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# boxes = target["boxes"].numpy()
# labels = target["labels"].numpy()
# for box, label in zip(boxes, labels):
#     x, y, w, h = box
#     x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
#     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#     cv2.putText(img, str(label), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
# plt.imshow(img)
# plt.axis('off')
# plt.show()
