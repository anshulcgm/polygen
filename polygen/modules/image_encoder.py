from typing import Dict, Optional, Tuple, List, Any

import torch
import torch.nn as nn
from torch.nn import Conv2d, Parameter, Dropout, ReLU, MaxPool2d
import torch.nn.functional as f
import torchvision.models as models
import pytorch_lightning as pl


class PolygenResnet(nn.Module):
    """Simple resnet used to extract image features"""

    def __init__(self) -> None:
        super(PolygenResnet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through first 3 resnet layers

        Args:
            x: Image tensor of shape [batch_size, num_channels, img_height, img_width]

        Returns:
            x: Image tensor after forward pass through first 3 layers of Resnet18
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)

        return x
