import os 
import torch 
from torch.utils.data import Dataset
import numpy as np 

"""
Dataset Loader

This file defines a PyTorch Dataset class for loading data in a clean and lightweight manner. 
Each data sample corresponds to one vehicle's perception at a single timestep.

For each sample, the dataset:
- Load LiDAR point cloud data
- Loads semantic labels (if available)
- Loads vehicle pose information
- Optionally loads neighbouring vehicle data

This dataset is designed for BEV-based semantic segmentation
and cooperative perception experiments, without relying on the OpenCOOD codebase.
"""

class OPV2VDataset(Dataset):

    def __init__ (self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split 
        self.transform = transform
        self.files = sorted(os.listdir(os.path.join(data_dir, split)))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path =  os.path.join(self.data_dir, self.split, self.files[idx])
        data = np.load(file_path, allow_pickle=True).item()
        # example keys: lidar, labels, pose, neighbor_data
        lidar = data['lidar']  # [n-points, 4]
        labels = data['labels']  # semantic labels
        pose = data['pose']      # vehicle pose
        neighbors = data.get('neighbor_data', None)  # optional 

        if self.transform: 
            lidar, labels = self.transform(lidar, labels)
        
        # convert to tensors return as dict
        return {
            'lidar': torch.tensor(lidar, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'pose': torch.tensor(pose, dtype=torch.float32),
            'neighbors': neighbors
        }
