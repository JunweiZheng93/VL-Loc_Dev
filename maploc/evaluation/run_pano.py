# Copyright (c) Meta Platforms, Inc. and affiliates.

import functools
from itertools import islice
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from torchmetrics import MetricCollection
from tqdm import tqdm

from .. import EXPERIMENTS_PATH, logger
from ..data.torch import collate, unbatch_to_device
from ..models.metrics_v1gt import AngleError, LateralLongitudinalError, Location2DError
from ..models.sequential import GPSAligner, RigidAligner
from ..models.voting import argmax_xyr, fuse_gps
from ..module import GenericModule
from ..utils.io import DATA_URL, download_file
from .utils import write_dump
from .viz import plot_example_sequential, plot_example_single

pretrained_models = dict(
    OrienterNet_MGL=("orienternet_mgl.ckpt", dict(num_rotations=256)),
)


def resolve_checkpoint_path(experiment_or_path: str) -> Path:
    path = Path(experiment_or_path)
    if not path.exists():
        # provided name of experiment
        path = Path(EXPERIMENTS_PATH, *experiment_or_path.split("/"))
        if not path.exists():
            if experiment_or_path in set(p for p, _ in pretrained_models.values()):
                download_file(f"{DATA_URL}/{experiment_or_path}", path)
            else:
                raise FileNotFoundError(path)
    if path.is_file():
        return path
    # provided only the experiment name
    maybe_path = path / "last-step.ckpt"
    if not maybe_path.exists():
        maybe_path = path / "step.ckpt"
    if not maybe_path.exists():
        raise FileNotFoundError(f"Could not find any checkpoint in {path}.")
    return maybe_path


@torch.no_grad()
def evaluate_single_image(
    dataloader: torch.utils.data.DataLoader,
    model: GenericModule,
    num: Optional[int] = None,
    callback: Optional[Callable] = None,
    progress: bool = True,
    mask_index: Optional[Tuple[int]] = None,
    has_gps: bool = False,
):
    """
    Evaluates a model on a dataset by processing images as panoramas (groups of 3).
    Note: The function now iterates directly over the dataset, not the dataloader,
    to preserve the order of panoramic views.
    """
    num_views = 3
    ppm = model.model.conf.pixel_per_meter
    metrics = MetricCollection(model.model.metrics())
    metrics["directional_error"] = LateralLongitudinalError(ppm)
    if has_gps:
        metrics["xy_gps_error"] = Location2DError("uv_gps", ppm)
        metrics["xy_fused_error"] = Location2DError("uv_fused", ppm)
        metrics["yaw_fused_error"] = AngleError("yaw_fused")
    metrics = metrics.to(model.device)

    dataset = dataloader.dataset
    num_panoramas = len(dataset) // num_views
    if num is not None:
        num_panoramas = min(num, num_panoramas)

    for i in tqdm(range(num_panoramas), total=num_panoramas, disable=not progress):
        start_idx = i * num_views
        # Manually create a batch of num_views for one panorama
        # batch_list = [dataset[start_idx], dataset[start_idx+1], dataset[start_idx+2], dataset[start_idx+3]]
        batch_list = [dataset[start_idx], dataset[start_idx+1], dataset[start_idx+2]]
        batch_ = collate(batch_list)
        batch = model.transfer_batch_to_device(batch_, model.device, i)

        # Ablation: mask semantic classes
        if mask_index is not None:
            # Apply mask to all num_views views in the panorama
            for view_idx in range(num_views):
                mask = batch["map"][view_idx, mask_index[0]] == (mask_index[1] + 1)
                batch["map"][view_idx, mask_index[0]][mask] = 0
        
        pred = model(batch)
        
        # In panorama mode, ground truth corresponds to the first view
        gt_data = batch_list[0]
        

        if has_gps:
            (uv_gps,) = pred["uv_gps"] = gt_data["uv_gps"][None] # Add batch dim
            pred["log_probs_fused"] = fuse_gps(
                pred["log_probs"], uv_gps, ppm, sigma=gt_data["accuracy_gps"]
            )
            uvt_fused = argmax_xyr(pred["log_probs_fused"])
            pred["uv_fused"] = uvt_fused[..., :2]
            pred["yaw_fused"] = uvt_fused[..., -1]
            del uv_gps, uvt_fused

        # The metrics have been updated to handle panorama logic via 'is_panorama' flag
        results = metrics(pred, batch) 
        if callback is not None:
            # For visualization, we typically want to show the main view (view 1)
            callback(
                i, model, unbatch_to_device(pred), unbatch_to_device(gt_data), results
            )
        del batch_, batch, pred, results

    return metrics.cpu()


@torch.no_grad()
def evaluate_sequential(
    dataset: torch.utils.data.Dataset,
    seq2idx: Dict,
    model: GenericModule,
    num: Optional[int] = None,
    shuffle: bool = False,
    callback: Optional[Callable] = None,
    progress: bool = True,
    num_rotations: int = 512,
    mask_index: Optional[Tuple[int]] = None,
    has_gps: bool = False,
):
    """Online (causal) sequential evaluation of each trajectory.

    Each key of seq2idx is one real trajectory (never mixed or chunked).
    Each frame's estimate (uv_seq/yaw_seq) is the argmax of the recursive
    belief right after fusing that frame's observation, i.e. it only uses
    past observations.
    """
    num_views = 3
    seq_keys = list(seq2idx)
    if shuffle:
        seq_keys = [seq_keys[i] for i in torch.randperm(len(seq_keys))]
    if num is not None:
        seq_keys = seq_keys[:num]
    lengths = [len(seq2idx[k]) // num_views for k in seq_keys] # Length in panoramas
    logger.info(
        "Min/max/med trajectory lengths: %d/%d/%d, total number of panoramas: %d",
        min(lengths),
        np.median(lengths),
        max(lengths),
        sum(lengths),
    )
    viz = callback is not None

    metrics = MetricCollection(model.model.metrics())
    ppm = model.model.conf.pixel_per_meter
    metrics["directional_error"] = LateralLongitudinalError(ppm)
    metrics["xy_seq_error"] = Location2DError("uv_seq", ppm)
    metrics["yaw_seq_error"] = AngleError("yaw_seq")
    metrics["directional_seq_error"] = LateralLongitudinalError(ppm, key="uv_seq")
    if has_gps:
        metrics["xy_gps_error"] = Location2DError("uv_gps", ppm)
        metrics["xy_gps_seq_error"] = Location2DError("uv_gps_seq", ppm)
        metrics["yaw_gps_seq_error"] = AngleError("yaw_gps_seq")
    metrics = metrics.to(model.device)

    keys_save = ["uvr_max", "uv_max", "yaw_max", "uv_expectation"]
    if has_gps:
        keys_save.append("uv_gps")
    if viz:
        keys_save.append("log_probs")

    for seq_index, key in enumerate(tqdm(seq_keys, disable=not progress)):
        indices = seq2idx[key]
        num_panoramas = len(indices) // num_views

        if num_panoramas == 0:
            continue

        aligner = RigidAligner()
        if has_gps:
            aligner_gps = GPSAligner(track_priors=viz, num_rotations=num_rotations)

        gt_batches = []
        preds = []
        uvt_online = []

        for i in range(num_panoramas):
            start_idx = i * num_views
            # Manually create a batch of num_views for one panorama
            batch_list = [dataset[indices[start_idx + v]] for v in range(num_views)]
            batch_ = collate(batch_list)

            # Ground truth and canvas are taken from the main view (view 1)
            gt_data = batch_list[0]

            batch = model.transfer_batch_to_device(batch_, model.device, 0)

            # Ablation: mask semantic classes on all views
            if mask_index is not None:
                for view_idx in range(num_views):
                    mask = batch["map"][view_idx, mask_index[0]] == (mask_index[1] + 1)
                    batch["map"][view_idx, mask_index[0]][mask] = 0

            pred = model(batch)

            canvas = gt_data["canvas"]
            gt_data["xy_geo"] = xy = canvas.to_xy(gt_data["uv"].double())
            gt_data["yaw"] = yaw = gt_data["roll_pitch_yaw"][-1].double()
            aligner.update_with_ref(pred["log_probs"][0], canvas, xy, yaw)
            uvt_online.append(argmax_xyr(aligner.belief))

            if has_gps:
                (uv_gps,) = pred["uv_gps"] = gt_data["uv_gps"][None]
                xy_gps = canvas.to_xy(uv_gps.double())
                aligner_gps.update(xy_gps, gt_data["accuracy_gps"], canvas, xy, yaw)

            if not viz:
                # To save memory, we can pop large tensors if not visualizing
                for item in batch_list:
                    item.pop("image", None)
                    item.pop("map", None)
            
            gt_batches.append(gt_data)
            preds.append({k: pred[k][0] for k in keys_save})
            del pred, batch, batch_

        if has_gps:
            xy_gt = torch.stack([b["xy_geo"] for b in gt_batches])
            yaw_gt = torch.stack([b["yaw"] for b in gt_batches])
            aligner_gps.compute()
            xy_gps_seq, yaw_gps_seq = aligner_gps.transform(xy_gt, yaw_gt)
            
        results = []
        for i in range(num_panoramas):
            preds[i]["uv_seq"] = uvt_online[i][:2].float()
            preds[i]["yaw_seq"] = uvt_online[i][-1].float()
            if has_gps:
                preds[i]["uv_gps_seq"] = (
                    gt_batches[i]["canvas"].to_uv(xy_gps_seq[i]).float()
                )
                preds[i]["yaw_gps_seq"] = yaw_gps_seq[i].float()
            
            # For metrics, we need to pass the full batch (for all 3 views)
            # but our metrics are already adapted to handle this via 'is_panorama'
            # and will internally select the correct GT.
            # Here we can just pass the GT of the main view for simplicity, as
            # the metrics will only use the keys from this dict.
            results.append(metrics(preds[i], gt_batches[i]))
            
        if viz:
            callback(seq_index, model, gt_batches, preds, results, aligner)
        del aligner, uvt_online, preds, gt_batches, results
        if has_gps:
            del aligner_gps

    return metrics.cpu()


def evaluate(
    experiment: str,
    cfg: DictConfig,
    dataset,
    split: str,
    sequential: bool = False,
    output_dir: Optional[Path] = None,
    callback: Optional[Callable] = None,
    num_workers: int = 1,
    viz_kwargs=None,
    **kwargs,
):
    if experiment in pretrained_models:
        experiment, cfg_override = pretrained_models[experiment]
        cfg = OmegaConf.merge(OmegaConf.create(dict(model=cfg_override)), cfg)

    logger.info("Evaluating model %s with config %s", experiment, cfg)
    checkpoint_path = resolve_checkpoint_path(experiment)
    model = GenericModule.load_from_checkpoint(
        checkpoint_path, cfg=cfg, find_best=not experiment.endswith(".ckpt")
    )
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    dataset.prepare_data()
    dataset.setup()

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        if callback is None:
            if sequential:
                callback = plot_example_sequential
            else:
                callback = plot_example_single
            callback = functools.partial(
                callback, out_dir=output_dir, **(viz_kwargs or {})
            )
    kwargs = {**kwargs, "callback": callback}

    seed_everything(dataset.cfg.seed)
    if sequential:
        dset, seq2idx = dataset.sequence_dataset(split)
        metrics = evaluate_sequential(dset, seq2idx, model, **kwargs)
    else:
        # Note: num_workers > 0 might cause issues if the dataset is not thread-safe.
        # For our new panorama logic, we use a dataloader just to access the dataset.
        loader = dataset.dataloader(split, shuffle=False, num_workers=num_workers)
        metrics = evaluate_single_image(loader, model, **kwargs)

    results = metrics.compute()
    logger.info("All results: %s", results)
    if output_dir is not None:
        write_dump(output_dir, experiment, cfg, results, metrics)
        logger.info("Outputs have been written to %s.", output_dir)
    return metrics