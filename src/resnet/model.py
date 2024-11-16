import torch
import torch.nn as nn
import torchvision


def load_model(model="resnet50"):
    return getattr(torchvision.models, model)

def forward(self, x):
  x = self.conv1(x)
  x = self.bn1(x)
  x = self.relu(x)
  x = self.maxpool(x)

  x = self.layer1(x)
  x = self.layer2(x)
  x = self.layer3(x)
  x = self.layer4(x)

  # We remove those two layers...
  # x = self.avgpool(x)
  # x = self.fc(x)

  # ... and we add this layer:
  x = self.conv1x1(x)

  return x

def detection_model(model="resnet50"):
    net = load_model("resnet50")
    net.forward = forward.__get__(
        net,
        torchvision.models.ResNet
    );  # monkey-patching
    conv1x1 = nn.Conv2d(2048, 1000, kernel_size=1)
    conv1x1.weight.data = net.fc.weight.data[..., None, None]
    conv1x1.bias.data = net.fc.bias.data
    net.conv1x1 = conv1x1

def main():
    net = load_model("resnet50")
    net.forward = forward.__get__(
        net,
        torchvision.models.ResNet
    );  # monkey-patching
    conv1x1 = nn.Conv2d(2048, 1000, kernel_size=1)


if __name__ == "__main__":
    main()
