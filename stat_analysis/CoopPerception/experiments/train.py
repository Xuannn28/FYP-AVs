import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.opv2v_dataset import OPV2VDataset
from models.bev_encoder import BEVEncoder
from models.seg_head import SegHead
from utils.metrics import compute_iou, MetricsTracker
from utils.visualization import plot_training_curves

"""
Training script for BEV semantic segmentation

This script trains the BEV encoder and semantic segmentation head
on the OPV2V dataset for cooperative perception experiments.
"""


def train_one_epoch(model_encoder, model_seg, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch

    Args:
        model_encoder: BEV encoder model
        model_seg: Semantic segmentation head
        dataloader: training data loader
        criterion: loss function
        optimizer: optimizer
        device: cuda or cpu
        epoch: current epoch number

    Returns:
        avg_loss: average loss for the epoch
    """
    model_encoder.train()
    model_seg.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Batch is a list of samples due to custom collate_fn
        for sample in batch:
            lidar = sample['lidar'].unsqueeze(0).to(device)  # [1, N, 4]
            labels = sample['labels'].to(device)  # [H, W]

            # Forward pass
            bev_features = model_encoder(lidar)  # [1, C, H, W]
            bev_logits = model_seg(bev_features)  # [1, num_classes, H, W]

            # Compute loss
            # Reshape for cross entropy: [B*H*W, C] and [B*H*W]
            logits_flat = bev_logits.permute(0, 2, 3, 1).reshape(-1, bev_logits.shape[1])
            labels_flat = labels.reshape(-1)

            loss = criterion(logits_flat, labels_flat)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def validate(model_encoder, model_seg, dataloader, criterion, device, num_classes=6):
    """
    Validate the model

    Args:
        model_encoder: BEV encoder model
        model_seg: Semantic segmentation head
        dataloader: validation data loader
        criterion: loss function
        device: cuda or cpu
        num_classes: number of semantic classes

    Returns:
        avg_loss: average validation loss
        mean_iou: mean IoU across all samples
    """
    model_encoder.eval()
    model_seg.eval()

    total_loss = 0.0
    num_batches = 0

    metrics_tracker = MetricsTracker(num_classes)

    with torch.no_grad():
        for batch in dataloader:
            for sample in batch:
                lidar = sample['lidar'].unsqueeze(0).to(device)
                labels = sample['labels'].to(device)

                # Forward pass
                bev_features = model_encoder(lidar)
                bev_logits = model_seg(bev_features)

                # Compute loss
                logits_flat = bev_logits.permute(0, 2, 3, 1).reshape(-1, bev_logits.shape[1])
                labels_flat = labels.reshape(-1)
                loss = criterion(logits_flat, labels_flat)

                total_loss += loss.item()
                num_batches += 1

                # Compute metrics
                predictions = torch.argmax(bev_logits, dim=1).squeeze(0)  # [H, W]
                metrics_tracker.update(predictions, labels)

    avg_loss = total_loss / max(num_batches, 1)
    summary = metrics_tracker.get_summary()
    mean_iou = summary['mean_iou']

    return avg_loss, mean_iou


def main():
    # =========================
    # Configuration
    # =========================
    DATA_DIR = "~/Downloads/OPV2V"
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    NUM_CLASSES = 6
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR = "checkpoints"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Training on device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")

    # =========================
    # Dataset and DataLoader
    # =========================
    train_dataset = OPV2VDataset(data_dir=DATA_DIR, split='train')
    val_dataset = OPV2VDataset(data_dir=DATA_DIR, split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: x
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # =========================
    # Models
    # =========================
    bev_encoder = BEVEncoder(
        bev_height=200,
        bev_width=200,
        num_features=64
    ).to(DEVICE)

    seg_head = SegHead(
        in_channels=64,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    # =========================
    # Loss and Optimizer
    # =========================
    # Use class weights to handle imbalanced classes
    # Typically background is most common, vehicles/pedestrians are rare
    class_weights = torch.tensor([0.5, 2.0, 3.0, 2.5, 1.0, 1.0]).to(DEVICE)  # Adjust based on dataset
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Combine parameters from both models
    optimizer = optim.Adam(
        list(bev_encoder.parameters()) + list(seg_head.parameters()),
        lr=LEARNING_RATE
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # =========================
    # Training loop
    # =========================
    train_losses = []
    val_losses = []
    val_ious = []
    best_iou = 0.0

    print("\n" + "="*50)
    print("Starting Training")
    print("="*50 + "\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")

        # Train
        train_loss = train_one_epoch(
            bev_encoder, seg_head, train_loader, criterion, optimizer, DEVICE, epoch
        )
        train_losses.append(train_loss)

        # Validate
        val_loss, val_iou = validate(
            bev_encoder, seg_head, val_loader, criterion, DEVICE, NUM_CLASSES
        )
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        # Learning rate scheduling
        scheduler.step()

        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val IoU: {val_iou:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': bev_encoder.state_dict(),
                'seg_head_state_dict': seg_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"  ✓ Saved best model (IoU: {best_iou:.4f})")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': bev_encoder.state_dict(),
                'seg_head_state_dict': seg_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint at epoch {epoch}")

    # =========================
    # Training complete
    # =========================
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best Validation IoU: {best_iou:.4f}")

    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, val_ious,
        save_path=os.path.join(CHECKPOINT_DIR, "training_curves.png")
    )
    print(f"Training curves saved to {CHECKPOINT_DIR}/training_curves.png")


if __name__ == "__main__":
    main()
