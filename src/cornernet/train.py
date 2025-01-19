# train on voc dataset

from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
import numpy as np

from model import CornerNet
from loss import CornerLoss


def collate_fn(batch):
    """custom collate function to handle variable sized images and annotations"""
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    # stack images after they've been resized by the transform
    images = torch.stack(images, 0)
    
    return images, targets

# output size is the size of the heatmaps and regrs
def process_target(target, num_classes, output_size=(128, 128)):
    """Convert VOC annotation to CornerNet targets"""

    # heatmaps are for localization of the top left and bottom right bouding box
    tl_heatmaps = torch.zeros((num_classes, output_size[0], output_size[1]))
    br_heatmaps = torch.zeros((num_classes, output_size[0], output_size[1]))

    # regrs are storing the regression targets for the top left and bottom right bouding box
    tl_regrs = torch.zeros((2, output_size[0], output_size[1]))
    br_regrs = torch.zeros((2, output_size[0], output_size[1]))

    # tags are used to indicate the presence of a corner
    tl_tags = torch.zeros((output_size[0], output_size[1]))
    br_tags = torch.zeros((output_size[0], output_size[1]))

    # mask is used to indicate the valid entries in the heatmaps and regrs
    mask = torch.zeros((output_size[0], output_size[1]))

    ann = target['annotation']
    w_ratio = output_size[1] / float(ann['size']['width'])
    h_ratio = output_size[0] / float(ann['size']['height'])

    for obj in ann['object']:
        bbox = obj['bndbox']
        cls_id = VOC_CLASSES.index(obj['name'])
        
        # convert coordinates to model output space
        xmin = float(bbox['xmin']) * w_ratio
        ymin = float(bbox['ymin']) * h_ratio
        xmax = float(bbox['xmax']) * w_ratio
        ymax = float(bbox['ymax']) * h_ratio
        
        # generate gaussian peaks for corners
        # make sure the model is able to learn the corners of the bounding box
        # the peak is refering to the real position of the corner in the heatmap
        tl_heatmaps[cls_id] = generate_gaussian(tl_heatmaps[cls_id], (int(xmin), int(ymin)))
        br_heatmaps[cls_id] = generate_gaussian(br_heatmaps[cls_id], (int(xmax), int(ymax)))
        
        # set regression targets and tags
        # the regression targets are the coordinates of the corner in the heatmap
        tl_regrs[0, int(ymin), int(xmin)] = xmin - int(xmin)
        tl_regrs[1, int(ymin), int(xmin)] = ymin - int(ymin)
        br_regrs[0, int(ymax), int(xmax)] = xmax - int(xmax)
        br_regrs[1, int(ymax), int(xmax)] = ymax - int(ymax)
        
        tl_tags[int(ymin), int(xmin)] = 1
        br_tags[int(ymax), int(xmax)] = 1
        mask[int(ymin), int(xmin)] = 1
        mask[int(ymax), int(xmax)] = 1

    return [tl_heatmaps, br_heatmaps, tl_tags, br_tags, tl_regrs, br_regrs, mask]

def generate_gaussian(heatmap, center, sigma=2.5):
    """Generate 2D gaussian peak centered at given point"""
    tmp_size = sigma * 3
    mu_x = center[0]
    mu_y = center[1]
    
    w, h = heatmap.shape[0:2]
    
    x = np.arange(0, w, 1, np.float32)
    y = np.arange(0, h, 1, np.float32)
    x, y = np.meshgrid(x, y)
    
    g = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2))
    return torch.max(torch.tensor(g), heatmap)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CornerNet(num_stacks=2, in_channels=3, num_classes=len(VOC_CLASSES))
    model = model.to(device)
    model.train()
    
    criterion = CornerLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    
    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.VOCDetection(
        root='./data',
        year='2012',
        image_set='train',
        download=True,
        transform=transform,
    )

    # img, target = dataset[0]
    # print(img.shape, target)
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            batch_targets = [process_target(target, len(VOC_CLASSES)) for target in targets]
            batch_targets = [torch.stack(x).to(device) for x in zip(*batch_targets)]
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss/len(dataloader):.4f}")
        
        # save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')

# VOC classes
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

if __name__ == '__main__':
    main()
