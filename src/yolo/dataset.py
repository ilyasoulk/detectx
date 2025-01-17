import lightning as L
import torch
from torchvision.datasets import VOCDetection
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader

CLASS_TO_IDX = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19,
    "background": 20,
}

class PascalVOCDataset(Dataset):
    def __init__(self, root, year="2012", image_set="train", transforms=None, img_dim=448, cell_dim=7, num_anchors=2, num_classes=20):
        self.voc_dataset = VOCDetection(
            root=root, year=year, image_set=image_set, download=True
        )
        self.transforms = transforms
        self.img_dim = img_dim
        self.cell_dim = cell_dim
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        
    def __len__(self):
        return len(self.voc_dataset)
    
    def parse_voc_annotation(self, annotation):
        boxes, labels = [], []
        objs = annotation["annotation"]["object"]
        if not isinstance(objs, list):
            objs = [objs]
            
        for obj in objs:
            bbox = obj["bndbox"]
            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])
            
            # Convert to center format (x, y, w, h)
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin
            
            boxes.append([x, y, w, h])
            labels.append(CLASS_TO_IDX[obj["name"]])
            
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        
    def create_yolo_target(self, target):
        target_tensor = torch.zeros(self.cell_dim, self.cell_dim, self.num_anchors * 5 + self.num_classes) # (7, 7, 2, 30)
        cell_size = self.img_dim / self.cell_dim
        
        for idx in range(len(target["boxes"])):
            x, y, w, h = target["boxes"][idx]
            label = target["labels"][idx]
            
            grid_x = int(x / cell_size)
            grid_y = int(y / cell_size)
            
            x_offset = (x % cell_size) / cell_size
            y_offset = (y % cell_size) / cell_size
            w_scale = w / self.img_dim
            h_scale = h / self.img_dim
            
            for anchor in range(self.num_anchors):
                anchor_start = anchor * 5
                target_tensor[grid_x, grid_y, anchor_start:anchor_start+5] = torch.tensor([x_offset, y_offset, w_scale, h_scale, 1.0])
            target_tensor[grid_x, grid_y, self.num_anchors * 5 + label] = 1.0
            
        return target_tensor

    def __getitem__(self, idx):
        image, target = self.voc_dataset[idx]
        target = self.parse_voc_annotation(target)
        
        if self.transforms:
            image = self.transforms(image)
            
        target = self.create_yolo_target(target)
        
        return image, target
