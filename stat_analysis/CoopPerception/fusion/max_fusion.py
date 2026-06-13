import torch 

"""
BEV Semantic Map Fusion

This module implements a simple max-based fusion strategy for 
cooperative perception. Given multiple BEV semantic maps from 
different vehicles, the fusion process selects the maximum class 
value at each BEV cell. 

Max fusion assumes: "higher class ID represent more important semantics."
e.g. 
Vehicle A      Vehicle B     Fused BEV map
0 0 0           0 0 0         0 0 0
0 2 0           0 0 3         0 2 3
0 0 0           0 0 0         0 0 0

This method assumes all BEV maps are spatially aligned in a common
coordinate frame and served as a lightweight baseline for cooperative 
semantic fusion with minimal computational cost.
"""
def max_fusion(bev_maps):
    """
    bev_maps: list of [H, W] semantic maps from neighbors
    Output: fused [H, W]
    """
    fused = bev_maps[0].clone()
    for bmap in bev_maps[1:]:
        fused = torch.max(fused, bmap)
    return fused
