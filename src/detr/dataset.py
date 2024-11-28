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
    def __init__(self, root, year="2007", image_set="train", transforms=None):
        self.voc_dataset = VOCDetection(
            root=root, year=year, image_set=image_set, download=True
        )
        self.transforms = transforms

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

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(
            labels, dtype=torch.int64
        )

    def __getitem__(self, idx):
        image, target = self.voc_dataset[idx]
        boxes, labels = self.parse_voc_annotation(target)
        width, height = image.size

        # Normalize bounding boxes to [0, 1]
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


class VOCTransforms:
    def __init__(self, train=True):
        self.train = train
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, image, target):
        # Resize
        image = F.to_tensor(image)
        image = F.resize(image, (512, 512))

        if self.train:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                image = F.hflip(image)
                target["boxes"][:, [0, 2]] = 1 - target["boxes"][:, [2, 0]]

            # Add more augmentations here as needed
            # Example: Random brightness/contrast
            if torch.rand(1) > 0.5:
                image = F.adjust_brightness(
                    image, brightness_factor=1.0 + 0.2 * (torch.rand(1) - 0.5)
                )

        image = self.normalize(image)
        return image, target


class PascalVOCDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 2,
        num_workers: int = 4,
        year: str = "2007",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.year = year

    def setup(self, stage=None):
        # Called on every GPU
        train_transforms = VOCTransforms(train=True)
        val_transforms = VOCTransforms(train=False)

        if stage == "fit" or stage is None:
            self.train_dataset = PascalVOCDataset(
                root=self.data_dir,
                year=self.year,
                image_set="train",
                transforms=train_transforms,
            )

            self.val_dataset = PascalVOCDataset(
                root=self.data_dir,
                year=self.year,
                image_set="val",
                transforms=val_transforms,
            )

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = [item[1]["labels"] for item in batch]
        boxes = [item[1]["boxes"] for item in batch]

        num_queries = 30
        num_cls = 21  # 20 classes + no object class
        batch_size = len(batch)
        padded_labels = torch.full((batch_size, num_queries), fill_value=(num_cls - 1))
        padded_boxes = torch.zeros(batch_size, num_queries, 4)

        for i in range(batch_size):
            num_objects = len(labels[i])
            num_objects = min(num_objects, num_queries)

            padded_labels[i][:num_objects] = labels[i][:num_objects]
            padded_boxes[i][:num_objects] = boxes[i][:num_objects]

        return images, (labels, boxes)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
