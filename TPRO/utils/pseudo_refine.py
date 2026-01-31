"""
Refine pseudo mask by shortest-path propagation from confident to uncertain regions.
"""
from __future__ import annotations

import heapq
import numpy as np
import torch
import torch.nn.functional as F

NEIGHBOR_4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]
NEIGHBOR_8 = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]


def _refine_one_image(
    cam_b: np.ndarray,
    label_b: np.ndarray,
    lam_low: float,
    lam_high: float,
    max_cost: float,
    neighbor: int,
) -> np.ndarray:
    """
    cam_b: (C, H, W) float. label_b: (C,) 0/1.
    Returns: (H, W) int64, values 0..C (0=bg, 1..C=class).
    """
    C, H, W = cam_b.shape
    cam_b = np.maximum(cam_b, 0.0)
    m = np.max(cam_b, axis=0)
    argmax_cam = np.argmax(cam_b, axis=0).astype(np.int32)

    certain_fg = m >= lam_high
    bg = m <= lam_low
    uncertain = ~certain_fg & ~bg

    out = np.zeros((H, W), dtype=np.int64)
    out[bg] = 0
    out[certain_fg] = argmax_cam[certain_fg] + 1

    neighbors = NEIGHBOR_8 if neighbor == 8 else NEIGHBOR_4

    dist_per_class = np.full((C, H, W), np.inf, dtype=np.float32)
    for c in range(C):
        if label_b[c] < 0.5:
            continue
        seeds = certain_fg & (argmax_cam == c)
        if not np.any(seeds):
            continue
        dist_c = np.full((H, W), np.inf, dtype=np.float32)
        dist_c[seeds] = 0.0
        heap = [(0.0, int(i), int(j)) for i, j in zip(*np.where(seeds))]
        heapq.heapify(heap)
        while heap:
            d, i, j = heapq.heappop(heap)
            if d > dist_c[i, j]:
                continue
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= H or nj < 0 or nj >= W:
                    continue
                edge = 1.0 - float(cam_b[c, ni, nj])
                if edge < 0:
                    edge = 0.0
                nd = d + edge
                if nd < dist_c[ni, nj]:
                    dist_c[ni, nj] = nd
                    heapq.heappush(heap, (nd, ni, nj))
        dist_per_class[c] = dist_c

    if np.any(uncertain):
        i_unc, j_unc = np.where(uncertain)
        for idx in range(len(i_unc)):
            i, j = int(i_unc[idx]), int(j_unc[idx])
            best_c = -1
            best_cost = np.inf
            for c in range(C):
                if label_b[c] < 0.5:
                    continue
                cost = dist_per_class[c, i, j]
                if cost <= max_cost and cost < best_cost:
                    best_cost = cost
                    best_c = c
            if best_c >= 0:
                out[i, j] = best_c + 1
            else:
                out[i, j] = argmax_cam[i, j] + 1 if m[i, j] > lam_low else 0

    return out


def refine_pseudo_by_shortest_path(
    cam_fuse: torch.Tensor,
    cls_label: torch.Tensor,
    lam_low: float,
    lam_high: float,
    max_cost: float,
    neighbor: int,
    num_cls: int,
) -> torch.Tensor:
    """
    cam_fuse: (B, C+1, H, W) or (B, C, H, W). cls_label: (B, C).
    Returns: (B, H, W) int64 on cpu.
    """
    B = cam_fuse.shape[0]
    if cam_fuse.shape[1] == num_cls + 1:
        cam_fuse = cam_fuse[:, 1:, :, :]
    cam_np = cam_fuse.detach().cpu().float().numpy()
    label_np = (cls_label.detach().cpu().numpy() > 0.5).astype(np.float64)
    out_list = []
    for b in range(B):
        refined = _refine_one_image(
            cam_np[b], label_np[b], lam_low, lam_high, max_cost, neighbor
        )
        out_list.append(refined)
    return torch.from_numpy(np.stack(out_list, axis=0)).long()


def refined_to_soft_mask(refined: torch.Tensor, num_cls: int) -> torch.Tensor:
    """
    refined: (B, H, W) int 0..num_cls (0=bg, 1..num_cls=class).
    Returns: (B, num_cls+1, H, W) float one-hot.
    """
    B, H, W = refined.shape
    device = refined.device
    soft = torch.zeros(B, num_cls + 1, H, W, device=device, dtype=torch.float32)
    for c in range(num_cls + 1):
        soft[:, c, :, :] = (refined == c).float()
    return soft
