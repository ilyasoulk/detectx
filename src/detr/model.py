import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def split_into_heads(Q, K, V, num_heads):
    Q = Q.reshape(Q.shape[0], Q.shape[1], num_heads, -1)
    K = K.reshape(K.shape[0], K.shape[1], num_heads, -1)
    V = V.reshape(V.shape[0], V.shape[1], num_heads, -1)
    return Q, K, V


def head_level_self_attention(Q, K, V):
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)
    d = Q.shape[-1]


    A = (Q @ K.transpose(-1, -2) / d**0.5).softmax(-1)
    attn_out = A @ V
    return attn_out.transpose(1, 2), A


def concat_heads(input_tensor):
  return input_tensor.flatten(-2, -1)

class TransformerEncoder(nn.Module):
    pass


class TransformerDecoder(nn.Module):
    pass



class DeTr(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove fully connected layers
    pass


if __name__ == "__main__":
    
# Define the data transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

# Download the COCO dataset (train split)
    dataset = datasets.CocoDetection(
        root='./data/train2017',  # It will create this directory
        annFile='./data/annotations/instances_train2017.json',
        transform=transform,
        download=True  # This will download the dataset automatically
    )

# Create a DataLoader
    data_loader = DataLoader(coco_dataset, batch_size=1, shuffle=True)

# Retrieve a sample and check its shape
    # for images, targets in data_loader:
    #     print(f"Image batch shape: {images[0].shape}")  # Check the image tensor shape
    #     print(f"Target sample: {targets[0]}")  # Print one target annotation
    #     break
    for images, targets in data_loader:
        image = images[0]  # Get the first (and only) image in the batch
        target = targets[0]  # Get the annotations for this image
        
        # Convert the image tensor to a NumPy array for visualization
        image_np = image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
        
        # Plot the image
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image_np)
        
        # Draw bounding boxes and labels
        for annotation in target:
            bbox = annotation['bbox']  # COCO uses [x, y, width, height]
            label = annotation['category_id']
            
            # Draw the bounding box
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add the label
            ax.text(
                bbox[0], bbox[1] - 5, str(label), color='red', fontsize=12, backgroundcolor='white'
            )
        
        plt.axis('off')
        plt.show()
        break  # Visualize only the first imagfor images, targets in data_loader:
