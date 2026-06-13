import torch.nn as nn

class SegHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, 1)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        return self.head(x)
