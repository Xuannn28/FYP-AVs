import torch
from torch.utils.data import DataLoader

# Dataset
from data.opv2v_dataset import OPV2VDataset

# Models
from models.bev_encoder import BEVEncoder
from models.seg_head import SegHead

# Utils
from utils.compression import quantize_bev, sparse_encode
from utils.transform import transform_bev
# Fusion
from fusion.max_fusion import max_fusion


def main():
    # =========================
    # 1. Basic setup
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = "~/Downloads/OPV2V"
    SPLIT = "test"
    BATCH_SIZE = 1

    print(f"Running on device: {device}")

    # =========================
    # 2. Load OPV2V dataset
    # =========================
    dataset = OPV2VDataset(
        data_dir=DATA_DIR,
        split=SPLIT
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda x: x  # OPV2V samples may have variable agents
    )

    # =========================
    # 3. Initialize models
    # =========================
    bev_encoder = BEVEncoder(
        bev_height=200,
        bev_width=200,
        num_features=64
    ).to(device)

    seg_head = SegHead(
        in_channels=64,
        num_classes=6
    ).to(device)

    bev_encoder.eval()
    seg_head.eval()

    # =========================
    # 4. Inference loop
    # =========================
    for step, batch in enumerate(dataloader):
        print(f"\nProcessing frame {step}")

        semantic_maps = []

        # Process each sample in the batch (typically batch_size=1 for OPV2V)
        for sample in batch:
            # ---- Process ego vehicle ----
            lidar = sample["lidar"].unsqueeze(0).to(device)
            pose = sample["pose"]  # Ego vehicle pose

            # BEV feature extraction
            bev_feat = bev_encoder(lidar)

            # BEV semantic segmentation
            bev_logits = seg_head(bev_feat)

            # Quantization (bandwidth reduction)
            bev_semantic = quantize_bev(bev_logits)  # [1, H, W]
            bev_semantic = bev_semantic.squeeze(0)

            # No need to transform ego vehicle (already in ego frame)
            semantic_maps.append(bev_semantic)

            # Estimate bandwidth for ego vehicle
            sparse = sparse_encode(bev_semantic.unsqueeze(0))
            bytes_sent = sparse[0].shape[0] * 3  # (x, y, class) as uint8
            print(f" Ego agent: sent ~{bytes_sent} bytes")

            # ---- Process neighboring vehicles ----
            if sample["neighbors"] is not None:
                for neighbor_id, neighbor_data in enumerate(sample["neighbors"]):
                    neighbor_lidar = torch.tensor(neighbor_data["lidar"], dtype=torch.float32).unsqueeze(0).to(device)
                    neighbor_pose = torch.tensor(neighbor_data["pose"], dtype=torch.float32)

                    # BEV feature extraction for neighbor
                    neighbor_bev_feat = bev_encoder(neighbor_lidar)

                    # BEV semantic segmentation for neighbor
                    neighbor_bev_logits = seg_head(neighbor_bev_feat)

                    # Quantization
                    neighbor_bev_semantic = quantize_bev(neighbor_bev_logits)  # [1, H, W]
                    neighbor_bev_semantic = neighbor_bev_semantic.squeeze(0)

                    # Transform neighbor's BEV to ego vehicle's frame
                    aligned_bev = transform_bev(neighbor_bev_semantic, neighbor_pose)
                    semantic_maps.append(aligned_bev)

                    # Estimate bandwidth for neighbor
                    sparse_neighbor = sparse_encode(neighbor_bev_semantic.unsqueeze(0))
                    bytes_sent_neighbor = sparse_neighbor[0].shape[0] * 3
                    print(f" Neighbor {neighbor_id}: sent ~{bytes_sent_neighbor} bytes")

        # =========================
        # 5. Cooperative fusion
        # =========================
        if len(semantic_maps) > 1:
            fused_map = max_fusion(semantic_maps)
            print(" Cooperative fusion performed")
        else:
            fused_map = semantic_maps[0]
            print(" Single-agent perception")

        # =========================
        # 6. (Optional) Local detection
        # =========================
        # This is where local 3D detection would run.
        # Detection results are NOT transmitted.
        #
        # detector(fused_map or lidar)

        if step >= 5:
            break


if __name__ == "__main__":
    main()
