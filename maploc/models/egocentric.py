"""Egocentric OSM utilities of the stage-A captions: the field-of-view wedge and the OSM class
labels seen from a pose.

Conventions (pinned against a trained model by maploc/dev/test_egocentric_labels.py):
  * map raster: row 0 = north, column 0 = west (Canvas.to_uv flips y);
  * yaw: degrees, clockwise from north (the template rotated by yaw is matched);
  * camera-frame BEV grid: x to the right, z forward, in metres, row 0 = far.
A BEV cell (x, z) seen from pose (u, v, yaw) lies at the map offset
  d = R(-yaw) (x, z) = (x cos + z sin, -x sin + z cos)   [east, north]
  u' = u + d_x * ppm,  v' = v - d_y * ppm.
"""
import numpy as np
import torch
import torch.nn.functional as F


def fov_mask(grid_xz: torch.Tensor, fov_deg: float = 120.0, z_max: float = 32.0) -> torch.Tensor:
    """Cells inside the horizontal field of view wedge and within z_max. [h, w]"""
    x, z = grid_xz[..., 0], grid_xz[..., 1]
    half = np.tan(np.deg2rad(fov_deg / 2))
    return (z > 0) & (z <= z_max) & (x.abs() <= z * half)


def egocentric_labels(
    map_raster: torch.Tensor, uv: torch.Tensor, yaw_deg: torch.Tensor,
    grid_xz: torch.Tensor, ppm: float,
) -> torch.Tensor:
    """Nearest-neighbour OSM class indices of the map seen from pose (uv, yaw)
    on the camera-frame grid. map_raster [B,3,H,W] long, uv [B,2] px, yaw [B] deg,
    grid_xz [h,w,2] m -> [B,3,h,w] long (0 = none / outside the map)."""
    psi = torch.deg2rad(yaw_deg.to(grid_xz))[:, None, None]
    c, s = torch.cos(psi), torch.sin(psi)
    x, z = grid_xz[None, ..., 0], grid_xz[None, ..., 1]
    dx = c * x + s * z
    dy = -s * x + c * z
    u = uv[:, 0, None, None].to(grid_xz) + dx * ppm
    v = uv[:, 1, None, None].to(grid_xz) - dy * ppm
    H, W = map_raster.shape[-2:]
    grid = torch.stack([u / (W - 1) * 2 - 1, v / (H - 1) * 2 - 1], -1)
    out = F.grid_sample(
        map_raster.float(), grid, mode="nearest", padding_mode="zeros", align_corners=True
    )
    return out.round().long()
