"""
Localization-informed Regularization (LIR): L_cre only.
Ma (multi-threshold from CAM) -> confident regions -> L_cre (patch-class contrast).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def localization_informed_regularization(
    patch_feat_4: torch.Tensor,
    class_tokens_4: torch.Tensor,
    cam4: torch.Tensor,
    cls_labels: torch.Tensor,
    lam_l: float,
    lam_h: float,
    tau: float,
    weight_cre: float,
):
    """
    patch_feat_4: (B, L, C), class_tokens_4: (num_cls, C) or (1, num_cls, C)
    cam4: (B, num_cls, H, W), cls_labels: (B, num_cls)
    Returns: (loss_lir, {'l_cre': l_cre})
    """
    if patch_feat_4 is None or class_tokens_4 is None:
        device = cam4.device
        return torch.tensor(0.0, device=device), {"l_cre": torch.tensor(0.0, device=device)}

    B, L, C = patch_feat_4.shape
    num_cls = class_tokens_4.shape[0] if class_tokens_4.dim() == 2 else class_tokens_4.shape[1]
    if class_tokens_4.dim() == 3:
        class_tokens_4 = class_tokens_4.squeeze(0)
    H, W = cam4.shape[2], cam4.shape[3]
    if L != H * W:
        return torch.tensor(0.0, device=patch_feat_4.device), {"l_cre": torch.tensor(0.0, device=patch_feat_4.device)}

    cam_sigmoid = torch.sigmoid(cam4)
    cam_max, cam_argmax = cam_sigmoid.max(dim=1)
    Ma = torch.full((B, H, W), 255, device=cam4.device, dtype=torch.long)
    Ma[cam_max <= lam_l] = 0
    Ma[cam_max >= lam_h] = cam_argmax[cam_max >= lam_h].long() + 1

    patch_feat_4 = F.normalize(patch_feat_4, dim=-1)
    class_tokens_4 = F.normalize(class_tokens_4.float(), dim=-1)

    confident = (Ma >= 1) & (Ma <= num_cls)
    if not confident.any():
        return torch.tensor(0.0, device=patch_feat_4.device), {"l_cre": torch.tensor(0.0, device=patch_feat_4.device)}

    flat_idx = torch.where(confident.reshape(B, -1))
    batch_idx = flat_idx[0]
    pos_idx = flat_idx[1]
    q = patch_feat_4[batch_idx, pos_idx]
    c_ids = Ma.reshape(B, -1)[batch_idx, pos_idx].long() - 1
    p_pos = class_tokens_4[c_ids]
    logits_all = q @ class_tokens_4.t() / tau
    l_cre_raw = F.cross_entropy(logits_all, c_ids, reduction="mean")
    l_cre = weight_cre * l_cre_raw
    return l_cre, {"l_cre": l_cre_raw.detach()}
