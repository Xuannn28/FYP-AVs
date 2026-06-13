
import torch 
import torch.nn as nn

"""
BEV feature encoder module 

This module defines a lightweight BEV encoder that maps LiDAR point 
cloud data into a fixed-size BEV feature representation. The BEV space
is defined on the ground plane (x-y) with a predefined resolution and 
spatial extent. 

In this prototype implementation, the voxelization / pillarization step
is left as placeholder, and a simple CNN backbone is applied to an empty
BEV feature grid. This design allows the overall BEV-based perception 
pipeline to be validated before integrating a full LiDAR encoding method 
such as PointPillars or voxel-based encoders. 
"""

class BEVEncoder(nn.Module):
    def __init__(self, bev_height=200, bev_width=200, num_features=64):
        super().__init__()
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.num_features = num_features
        # Simple cnn after voxelization
        self.cnn = nn.Sequential(
            nn.Conv2d(num_features, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_features, 3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, lidar_points):
        B = lidar_points.shape[0]
        bev = torch.zeros(B, self.num_features, self.bev_height, self.bev_width).to(lidar_points.device)

        # TODO: implement voxelization / pillarization
        # for prototype, just return zeros 
        bev = self.cnn(bev)

        return bev