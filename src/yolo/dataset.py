import torch
from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

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
    def __init__(self, root, year="2012", image_set="train", transform=None, split_size=7, num_boxes=2, num_classes=20):
        self.voc_dataset = VOCDetection(
            root=root, year=year, image_set=image_set, download=True
        )
        self.transform = transform
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        
        
    def __len__(self):
        return len(self.voc_dataset)
    
    def parse_voc_annotation(self, annotation):
        boxes = []
        
        width = float(annotation["annotation"]["size"]["width"])
        height = float(annotation["annotation"]["size"]["height"])
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
            x = ((xmin + xmax) / 2) / width
            y = (ymin + ymax) / 2 / height
            w = xmax - xmin / width
            h = ymax - ymin / height
            
            class_idx = CLASS_TO_IDX[obj["name"]]
            
            boxes.append([class_idx, x, y, w, h])
            
        return boxes
        
    def create_yolo_target(self, target):
        target_tensor = torch.zeros(self.S, self.S, 5 * self.B + self.C) # (7, 7, 30) but we consider its (7, 7, 25)
        
        for idx in range(len(target)):
            class_label, x, y, w, h = target[idx]
            
            j = int(x * self.S)
            i = int(y * self.S)
            
            x_offset = x * self.S - j
            y_offset = y * self.S - i
            w_scale = w * self.S
            h_scale = h * self.S
            
            target_tensor[i, j, self.C:self.C + 5] = torch.tensor([1, x_offset, y_offset, w_scale, h_scale])
            target_tensor[i, j, class_label] = 1
            
        return target_tensor

    def __getitem__(self, idx):
        image, target = self.voc_dataset[idx]
        target = self.parse_voc_annotation(target)
        
        if self.transform:
            image = self.transform(image)
            
        target = self.create_yolo_target(target)
        
        return image, target
