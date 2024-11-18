from torchvision import datasets
from torchvision.transforms import transforms
import torch

# Define basic transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download PASCAL VOC 2012
# Set download=True to download if not already present
# By default downloads to ./data/VOCdevkit/VOC2012/
dataset = datasets.VOCDetection(
    root='./data',  # where to save the dataset
    year='2012',    # can use '2007' or '2012'
    image_set='train',  # can be 'train', 'val', or 'trainval'
    download=True,
    transform=transform
)

# Get a single example
img, target = dataset[0]
print(img.shape, target)

# target is a dict containing:
# - annotation: contains bounding boxes and object classes
# - image: image filename
