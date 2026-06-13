import torch
import torch.nn.functional as F
import math

def transform_bev(bev_map, pose, bev_resolution=0.5):
    """
    bev_map: [H, W]
    pose: [x, y, yaw] in meters / radians
    Returns: aligned bev_map in ego/global frame
    """
    H, W = bev_map.shape
    device = bev_map.device

    x, y, yaw = pose
    dx = x / bev_resolution
    dy = y / bev_resolution

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    theta = torch.tensor([
        [ cos_yaw, -sin_yaw, dx / W * 2],
        [ sin_yaw,  cos_yaw, dy / H * 2]
    ], device=device).unsqueeze(0)

    grid = F.affine_grid(theta, size=(1, 1, H, W), align_corners=False)
    bev = bev_map.unsqueeze(0).unsqueeze(0).float()
    aligned = F.grid_sample(bev, grid, align_corners=False)

    return aligned.squeeze()
