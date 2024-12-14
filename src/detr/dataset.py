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

            boxes.append([xmin, ymin, xmax, ymax])
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
    def __init__(self, train=True, normalize=True):
        self.train = train
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.do_normalize = normalize
        self.target_size = (512, 512)

    def __call__(self, image, target):
        # Get original dimensions
        orig_width, orig_height = image.size

        # Convert image to tensor
        image = F.to_tensor(image)

        # Resize image
        image = F.resize(image, self.target_size)

        # Adjust box coordinates for the new size
        boxes = target["boxes"]
        # Convert from normalized to absolute coordinates
        boxes[:, [0, 2]] *= orig_width  # x coordinates (xmin, xmax)
        boxes[:, [1, 3]] *= orig_height  # y coordinates (ymin, ymax)

        # Scale boxes to new dimensions
        scale_x = self.target_size[1] / orig_width
        scale_y = self.target_size[0] / orig_height

        boxes[:, [0, 2]] *= scale_x  # scale x coordinates
        boxes[:, [1, 3]] *= scale_y  # scale y coordinates

        # Normalize boxes to [0, 1]
        boxes[:, [0, 2]] /= self.target_size[1]  # x coordinates
        boxes[:, [1, 3]] /= self.target_size[0]  # y coordinates

        target["boxes"] = boxes

        if self.train:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                image = F.hflip(image)
                # For xmin, xmax format, we need to flip both coordinates
                boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]  # Flip and swap xmin, xmax
                target["boxes"] = boxes

            # Add more augmentations here as needed
            if torch.rand(1) > 0.5:
                image = F.adjust_brightness(
                    image, brightness_factor=1.0 + 0.2 * (torch.rand(1) - 0.5)
                )

        if self.do_normalize:
            image = self.normalize(image)

        return image, target


class PascalVOCDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 2,
        num_workers: int = 4,
        year: str = "2007",
        normalize: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.year = year
        self.normalize = normalize

    def setup(self, stage=None):
        # Called on every GPU
        train_transforms = VOCTransforms(train=True, normalize=self.normalize)
        val_transforms = VOCTransforms(train=False, normalize=self.normalize)

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
    def collate_fn(batch, num_queries=30):
        images = torch.stack([item[0] for item in batch])
        batch_size = len(batch)

        # Initialize lists to store padded tensors
        padded_labels_list = []
        padded_boxes_list = []

        for i in range(batch_size):
            labels = batch[i][1]["labels"]
            boxes = batch[i][1]["boxes"]

            # Get number of objects (limited by num_queries)
            num_objects = min(len(labels), num_queries)

            # Create padded tensor for this batch item
            padded_labels = torch.full(
                (num_queries,), fill_value=20
            )  # 20 is background class
            padded_boxes = torch.zeros((num_queries, 4))

            # Fill with actual values
            padded_labels[:num_objects] = labels[:num_objects]
            padded_boxes[:num_objects] = boxes[:num_objects]

            # Add to lists
            padded_labels_list.append(padded_labels)
            padded_boxes_list.append(padded_boxes)

        return images, (padded_labels_list, padded_boxes_list)

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
