from dataset import PascalVOCDataset
from model import YOLOv1
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb_track = False

if wandb_track:
    wandb.init(
        project="object-detection",
        
        config={
            "learning_rate": 0.001,
            "architecture": "YOLOv1",
            "dataset": "PASCAL VOC 2012",
            "epochs": 2,
        },
    )

# Define basic transformations, resize to 224x224
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

# Load the dataset
dataset = PascalVOCDataset(
    root='../../data',  # where to save the dataset
    year='2012',    # can use '2007' or '2012'
    image_set='train',  # can be 'train', 'val', or 'trainval'
    transforms=transform
)

# Split the dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
)


# Define the model
model = YOLOv1(num_classes=20).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 2

# Train the model, print loss only after epoch
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    if wandb_track:
        wandb.log({"train/epoch_loss": loss.item()})
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            output = model(images)
            loss = criterion(output, targets)
        print(f"Validation Loss: {loss.item()}")
        if wandb_track:
            wandb.log({"val/loss": loss.item()})

if wandb_track:
    wandb.finish()

# show a random image from the dataset, with bounding boxes
idx = np.random.randint(len(dataset))
img, target = dataset[idx]
img = img.permute(1, 2, 0).numpy()
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
boxes = target["boxes"].numpy()
labels = target["labels"].numpy()
for box, label in zip(boxes, labels):
    x, y, w, h = box
    x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img, str(label), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
plt.imshow(img)
plt.axis('off')
plt.show()
