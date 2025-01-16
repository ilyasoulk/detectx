import torch
import torchvision.transforms as T
from torchvision import ops
from torchvision.datasets import VOCDetection
import matplotlib.pyplot as plt

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
    "empty",
]

COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]
COLORS *= 100

revert_normalization = T.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


def parse_voc_annotation(target):
    # Get image size from the annotation dictionary
    width = float(target["annotation"]["size"]["width"])
    height = float(target["annotation"]["size"]["height"])

    boxes = []
    classes = []

    # Parse each object
    objects = target["annotation"].get("object", [])
    # Handle case where there's only one object (not in a list)
    if not isinstance(objects, list):
        objects = [objects]

    for obj in objects:
        # Get class id
        class_name = obj["name"]
        if class_name in CLASSES:
            class_id = CLASSES.index(class_name)

            # Get bounding box coordinates
            bbox = obj["bndbox"]
            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])

            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(class_id)

    if not boxes:  # If no valid objects were found
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros(0, dtype=torch.int64),
            width,
            height,
        )

    return (
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(classes, dtype=torch.int64),
        width,
        height,
    )


def plot_im_with_boxes(im, boxes, probs=None, ax=None):
    if ax is None:
        plt.imshow(im)
        ax = plt.gca()

    for i, b in enumerate(boxes.tolist()):
        xmin, ymin, xmax, ymax = b

        patch = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            color=COLORS[i],
            linewidth=2,
        )

        ax.add_patch(patch)
        if probs is not None:
            if probs.ndim == 1:
                cl = probs[i].item()
                text = f"{CLASSES[cl]}"
            else:
                cl = probs[i].argmax().item()
                text = f"{CLASSES[cl]}: {probs[i,cl]:0.2f}"
        else:
            text = ""

        ax.text(xmin, ymin, text, fontsize=7, bbox=dict(facecolor="yellow", alpha=0.5))


class MyVOCDetection(VOCDetection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge = 512

        self.T = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.Resize((self.edge, self.edge), antialias=True),
            ]
        )

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        # PIL image
        w, h = img.size

        input_ = self.T(img)

        # Parse VOC annotation
        boxes, classes, _, _ = parse_voc_annotation(target)

        # Convert boxes to relative coordinates and cxcywh format
        if len(boxes) > 0:  # Only if we have boxes
            boxes[:, 0::2] /= w  # scale x coordinates
            boxes[:, 1::2] /= h  # scale y coordinates
            boxes.clamp_(min=0, max=1)
            boxes = ops.box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")

        return input_, (classes, boxes)


def collate_fn(inputs):
    input_ = torch.stack([i[0] for i in inputs])
    classes = tuple([i[1][0] for i in inputs])
    boxes = tuple([i[1][1] for i in inputs])
    return input_, (classes, boxes)
