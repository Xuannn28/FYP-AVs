
import torch

def quantize_bev(bev_logits):
    """
    Convert float logits to uint8 semantic map
    Input: [B, C, H, W]
    Output: [B, H, W] class indices
    """
    # Take argmax to get semantic class
    bev_classes = torch.argmax(bev_logits, dim=1).byte()
    return bev_classes

def sparse_encode(bev_classes):
    """
    Encode non-zero cells only for transmission
    Returns list of (x, y, class) tuples
    """
    B, H, W = bev_classes.shape
    encoded = []
    for b in range(B):
        nonzero = (bev_classes[b] != 0).nonzero(as_tuple=False)
        values = bev_classes[b][bev_classes[b] != 0]
        encoded.append(torch.cat([nonzero, values.unsqueeze(1)], dim=1))
    return encoded
