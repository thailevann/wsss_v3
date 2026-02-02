"""
CAM-Guided Residual Refinement for Stage 3.
Refines mid-level features (t3) using CAM guidance without adding branches or losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CAMGuidedResidual(nn.Module):
    """
    Refines features using CAM-guided residual.
    Formula: feat_refined = feat + gamma * cam_mask * feat
    
    Args:
        init_gamma: Initial value for gamma parameter (default: 0.0 to start disabled)
    """
    def __init__(self, init_gamma=0.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))

    def forward(self, feat, cam_mask):
        """
        Args:
            feat: (B, C, H, W) - Raw features to refine
            cam_mask: (B, 1, H, W) - CAM mask in [0,1], should be detached
        Returns:
            feat_refined: (B, C, H, W) - Refined features
        """
        # cam_mask should be detached to avoid backprop through CAM
        return feat + self.gamma * cam_mask * feat


def normalize_cam(cam):
    """
    Normalize CAM to [0, 1] range.
    
    Args:
        cam: (B, num_cls, H, W) - Raw CAM
    Returns:
        cam_norm: (B, num_cls, H, W) - Normalized CAM in [0, 1]
    """
    cam = torch.clamp(cam, min=0)
    cam_max = cam.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    cam_min = cam.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-6)
    return cam_norm


def create_cam_mask(cam, detach=True):
    """
    Create foreground CAM mask from multi-class CAM.
    
    Args:
        cam: (B, num_cls, H, W) - Multi-class CAM
        detach: Whether to detach CAM (default: True to avoid backprop)
    Returns:
        cam_mask: (B, 1, H, W) - Foreground mask in [0, 1]
    """
    cam_norm = normalize_cam(cam)
    cam_fg = cam_norm.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
    if detach:
        cam_fg = cam_fg.detach()
    return cam_fg
