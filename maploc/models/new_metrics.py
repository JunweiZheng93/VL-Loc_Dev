# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torchmetrics
from torchmetrics.utilities.data import dim_zero_cat

from .utils import deg2rad, rotmat2d


def location_error(uv, uv_gt, ppm=1):
    return torch.norm(uv - uv_gt.to(uv), dim=-1) / ppm


def angle_error(t, t_gt):
    error = torch.abs(t % 360 - t_gt.to(t) % 360)
    error = torch.minimum(error, 360 - error)
    return error


class Location2DRecall(torchmetrics.MeanMetric):
    def __init__(self, threshold, pixel_per_meter, key="uv_max", *args, **kwargs):
        self.threshold = threshold
        self.ppm = pixel_per_meter
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        # --- FIX START ---
        is_panorama = pred.get("is_panorama", False)
        uv_gt = data["uv"]
        if is_panorama:
            uv_gt = uv_gt[::3]
        # --- FIX END ---
        error = location_error(pred[self.key], uv_gt, self.ppm)
        super().update((error <= self.threshold).float())

class AngleRecall(torchmetrics.MeanMetric):
    def __init__(self, threshold, key="yaw_max", *args, **kwargs):
        self.threshold = threshold
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        is_panorama = pred.get("is_panorama", False)
        yaw_pred = pred[self.key]

        if is_panorama:
            # Shape of yaw_pred: (N,) where N is the batch size
            # Shape: (3, N)
            yaw_gt_candidates = torch.stack([
                data["roll_pitch_yaw"][::3, -1],
                data["roll_pitch_yaw"][1::3, -1],
                data["roll_pitch_yaw"][2::3, -1],
            ], dim=0)

            # 为了利用广播机制，我们需要扩展yaw_pred的维度
            # (N,) -> (1, N)
            yaw_pred_expanded = yaw_pred.unsqueeze(0)

            # 一次性计算所有候选yaw的误差
            # 输出 Shape: (3, N)
            errors = angle_error(yaw_pred_expanded, yaw_gt_candidates)
            
            # 找到每个数据点对应的最小误差
            # Shape: (N,)
            min_error, _ = torch.min(errors, dim=0)
            
            error = min_error
        else:
            # 非全景图的原始逻辑
            yaw_gt = data["roll_pitch_yaw"][..., -1]
            error = angle_error(yaw_pred, yaw_gt)
        
        super().update((error <= self.threshold).float())



class MeanMetricWithRecall(torchmetrics.Metric):
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("value", default=[], dist_reduce_fx="cat")

    def compute(self):
        return dim_zero_cat(self.value).mean(0)

    def get_errors(self):
        return dim_zero_cat(self.value)

    def recall(self, thresholds):
        error = self.get_errors()
        thresholds = error.new_tensor(thresholds)
        return (error.unsqueeze(-1) < thresholds).float().mean(0) * 100


class AngleError(MeanMetricWithRecall):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def update(self, pred, data):
        # --- FIX START ---
        is_panorama = pred.get("is_panorama", False)
        yaw_pred = pred[self.key]
        
        if is_panorama:
            yaw_gt_v1 = data["roll_pitch_yaw"][::3, -1]
            yaw_gt_v2 = data["roll_pitch_yaw"][1::3,-1]
            yaw_gt_v3 = data["roll_pitch_yaw"][2::3,-1]
            yaw_gt_candidates = torch.stack([yaw_gt_v1, yaw_gt_v2, yaw_gt_v3], dim=0)
            yaw_pred_expanded = yaw_pred.unsqueeze(0)
            values = angle_error(yaw_pred_expanded, yaw_gt_candidates)
            min_value, _ = torch.min(values, dim=0)
            value = min_value
        else:
            yaw_gt = data["roll_pitch_yaw"][..., -1]
            value = angle_error(yaw_pred, yaw_gt)
        # --- FIX END ---
        if value.numel():
            self.value.append(value)


class Location2DError(MeanMetricWithRecall):
    def __init__(self, key, pixel_per_meter):
        super().__init__()
        self.key = key
        self.ppm = pixel_per_meter

    def update(self, pred, data):
        # --- FIX START ---
        is_panorama = pred.get("is_panorama", False)
        uv_gt = data["uv"]
        if is_panorama:
            uv_gt = uv_gt[::3]
        # --- FIX END ---
        value = location_error(pred[self.key], uv_gt, self.ppm)
        if value.numel():
            self.value.append(value)

class LateralLongitudinalError(MeanMetricWithRecall):
    def __init__(self, pixel_per_meter, key="uv_max"):
        super().__init__()
        self.ppm = pixel_per_meter
        self.key = key

    def update(self, pred, data):
        is_panorama = pred.get("is_panorama", False)
        uv_pred = pred[self.key]

        if is_panorama:
            # For panorama, we have 3 possible ground truth yaws. We pick the one
            # that yields the minimum error.
            uv_gt = data["uv"][::3]
            yaw_gt_candidates = torch.stack([
                data["roll_pitch_yaw"][::3, -1],
                data["roll_pitch_yaw"][1::3, -1],
                data["roll_pitch_yaw"][2::3, -1],
            ], dim=0) # Shape: (3, N) where N is the batch size

            # Calculate errors for all yaw candidates at once for efficiency
            yaw_candidates_rad = deg2rad(yaw_gt_candidates) # Shape: (3, N)

            # Expand dimensions for broadcasting: (N, 2) -> (1, N, 2)
            shift = (uv_pred.unsqueeze(0) - uv_gt.unsqueeze(0)) * uv_pred.new_tensor([-1, 1])

            # Perform batch rotation for all candidates
            # (3, N) -> (3*N,), then create rotation matrices (3*N, 2, 2)
            rot_matrices = rotmat2d(yaw_candidates_rad.flatten())
            # (1, N, 2) -> (3, N, 2) -> (3*N, 2, 1)
            expanded_shift = shift.expand(3, -1, -1).reshape(-1, 2, 1)

            # (3*N, 2, 2) @ (3*N, 2, 1) -> (3*N, 2, 1) -> (3*N, 2)
            rotated_shift = (rot_matrices @ expanded_shift).squeeze(-1)
            
            # Reshape back to (3, N, 2) to separate candidates
            error = (torch.abs(rotated_shift) / self.ppm).view(3, -1, 2)

            # Sum lateral and longitudinal errors for each candidate: (3, N)
            sum_errors = error.sum(dim=-1)
            
            # Find the index of the best yaw for each item in the batch: (N,)
            _, min_indices = torch.min(sum_errors, dim=0)
            
            # Gather the error values corresponding to the best yaw for each item
            # Permute error to (N, 3, 2) to easily index by batch
            value = error.permute(1, 0, 2)[torch.arange(len(min_indices)), min_indices]

        else:
            # Original logic for non-panorama images
            uv_gt = data["uv"]
            yaw_gt = data["roll_pitch_yaw"][..., -1]
            yaw = deg2rad(yaw_gt)
            shift = (uv_pred - uv_gt) * yaw.new_tensor([-1, 1])
            shift = (rotmat2d(yaw) @ shift.unsqueeze(-1)).squeeze(-1)
            error = torch.abs(shift) / self.ppm
            value = error.view(-1, 2)
        
        if value.numel():
            self.value.append(value)
