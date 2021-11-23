from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class SimpleConvNet(nn.Module):
    """Simple one-layer CNN to be used for testing."""

    def __init__(self, img_size: Tuple[int] = (144, 144), num_filters: int = 32, num_classes: int = 2):
        """Initialize model layers.

        Args:
            img_size: (Tuple[int]): Size of input images.
            num_filters (int): Number of filters in conv layer.
            num_classes (int): Number of classes to predict.
        """
        super(SimpleConvNet, self).__init__()
        self.conv = nn.Conv2d(3, num_filters, 3, padding="same")
        self.fc = nn.Linear(
            int((img_size[0]/2)*(img_size[1]/2)*num_filters), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x = self.conv(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
