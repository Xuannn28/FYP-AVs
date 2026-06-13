import torch
from data.opv2v_dataset import OPV2VDataset
from models.bev_encoder import BEVEncoder
from models.seg_head import SegHead          # fixed: was BEVSegHead
from utils.compression import quantize_bev, sparse_encode
from fusion.max_fusion import max_fusion     # fixed: removed CoopPerception prefix

"""
BEV semantic segmentation inference 

This script demonstrates a basic inference pipeline for cooperatie perception.
It performs the following steps:

1. Loads LiDAR data from the dataset.
2. Converts LiDAR point clouds into BEV feature maps. 
3. Predicts BEV semantic segmentation using a lightweight segmentation head.
4. Applies quantization and sparse encoding to reduce communication bandwidth. 
5. Optionally fuses BEV semantic maps from neighbouring vehicles. 

This script served as a prototype for validating the BEV-based semantic sharing 
approach and does not include training or full multi-agent synchronization.
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset
dataset = OPV2VDataset(data_dir='~/Downloads/OPV2V', split='test')
loader = torch.utils.data.DataLoader(dataset, batch_size=2)  # need custom collate_fn

# Load models
bev_encoder = BEVEncoder().to(device)
seg_head = SegHead(in_channels=64, num_classes=6).to(device)  # fixed: use SegHead with correct args

for batch in loader:
    lidar = batch['lidar'].to(device)
    # BEV features
    bev_feat = bev_encoder(lidar)
    # BEV semantic logits
    bev_logits = seg_head(bev_feat)
    # Quantize + sparse encode
    bev_classes = quantize_bev(bev_logits)
    sparse = sparse_encode(bev_classes)
    # Example: fuse with neighbors if any
    if batch['neighbors'] is not None:
        neighbor_maps = [torch.tensor(n).to(device) for n in batch['neighbors']]
        fused = max_fusion([bev_classes[0]] + neighbor_maps)
