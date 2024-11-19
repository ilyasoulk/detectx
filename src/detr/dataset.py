import torch
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader

CLASS_TO_IDX = {
    "aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4,
    "bus": 5, "car": 6, "cat": 7, "chair": 8, "cow": 9,
    "diningtable": 10, "dog": 11, "horse": 12, "motorbike": 13, "person": 14,
    "pottedplant": 15, "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 19,
    "background": 20
}

class PascalVOCDataset(Dataset):
    def __init__(self, root, year='2007', image_set='train', transforms=None):
        self.voc_dataset = VOCDetection(root=root, year=year, image_set=image_set, download=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.voc_dataset)

    def parse_voc_annotation(self, annotation):
        boxes, labels = [], []
        objs = annotation['annotation']['object']
        if not isinstance(objs, list):
            objs = [objs]

        for obj in objs:
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[obj['name']])

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        image, target = self.voc_dataset[idx]
        boxes, labels = self.parse_voc_annotation(target)
        width, height = image.size

        # Normalize bounding boxes to [0, 1]
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return F.to_tensor(image), target


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return torch.stack(images), targets


def transform(image, target):
    image = F.resize(image, (512, 512))  # Resize image
    if torch.rand(1) > 0.5:
        image = F.hflip(image)  # Horizontal flip

        # Flip bounding boxes
        target['boxes'][:, [0, 2]] = 1 - target['boxes'][:, [2, 0]]

    return image, target


def get_pascal_voc_dataloader(root, batch_size=8, num_workers=4, year='2007', image_set='train'):
    dataset = PascalVOCDataset(root=root, year=year, image_set=image_set, transforms=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader

