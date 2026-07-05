# Copyright (c) Meta Platforms, Inc. and affiliates.

import functools
from itertools import islice
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from torchmetrics import MetricCollection
from tqdm import tqdm

from .. import EXPERIMENTS_PATH, logger
from ..data.torch import collate, unbatch_to_device
from ..models.metrics import AngleError, LateralLongitudinalError, Location2DError
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
    gpus: int = 1,
):
    is_distributed = gpus > 1
    rank = dist.get_rank() if is_distributed else 0
    module = model.module if is_distributed else model
    ppm = module.model.conf.pixel_per_meter
    metrics = MetricCollection(module.model.metrics())
    metrics["directional_error"] = LateralLongitudinalError(ppm)
    if has_gps:
        metrics["xy_gps_error"] = Location2DError("uv_gps", ppm)
        metrics["xy_fused_error"] = Location2DError("uv_fused", ppm)
        metrics["yaw_fused_error"] = AngleError("yaw_fused")
    metrics = metrics.to(module.device)

    dataset = dataloader.dataset
    
    sampler = None
    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False, drop_last=False)

    # Recreate dataloader with the sampler
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=dataloader.batch_size or 1, 
        sampler=sampler, 
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=True
    )

    if rank == 0:
        pbar = tqdm(total=len(loader), disable=not progress)

    for i, batch in enumerate(loader):
        if num is not None and i >= num:
            break
        
        batch = module.transfer_batch_to_device(batch, module.device, i)

        with autocast(enabled=True):
            # Ablation: mask semantic classes
            if mask_index is not None:
                mask = batch["map"][0, mask_index[0]] == (mask_index[1] + 1)
                batch["map"][0, mask_index[0]][mask] = 0
            pred = model(batch)

            if has_gps:
                (uv_gps,) = pred["uv_gps"] = batch["uv_gps"]
                pred["log_probs_fused"] = fuse_gps(
                    pred["log_probs"], uv_gps, ppm, sigma=batch["accuracy_gps"]
                )
                uvt_fused = argmax_xyr(pred["log_probs_fused"])
                pred["uv_fused"] = uvt_fused[..., :2]
                pred["yaw_fused"] = uvt_fused[..., -1]
                del uv_gps, uvt_fused

        results = metrics(pred, batch)
        if callback is not None and rank == 0:
            callback(
                i, module, unbatch_to_device(pred), unbatch_to_device(gt_data), results
            )
        
        if rank == 0:
            pbar.update(1)

        del batch, pred, results
    
    if rank == 0:
        pbar.close()

    return metrics


@torch.no_grad()
def evaluate_sequential(
    dataset: torch.utils.data.Dataset,
    chunk2idx: Dict,
    model: GenericModule,
    num: Optional[int] = None,
    shuffle: bool = False,
    callback: Optional[Callable] = None,
    progress: bool = True,
    num_rotations: int = 512,
    mask_index: Optional[Tuple[int]] = None,
    has_gps: bool = False,
):
    module = model.module if hasattr(model, "module") else model
    chunk_keys = list(chunk2idx)
    if shuffle:
        chunk_keys = [chunk_keys[i] for i in torch.randperm(len(chunk_keys))]
    if num is not None:
        chunk_keys = chunk_keys[:num]
    lengths = [len(chunk2idx[k]) for k in chunk_keys]
    logger.info(
        "Min/max/med lengths: %d/%d/%d, total number of images: %d",
        min(lengths),
        np.median(lengths),
        max(lengths),
        sum(lengths),
    )
    viz = callback is not None

    metrics = MetricCollection(module.model.metrics())
    ppm = module.model.conf.pixel_per_meter
    metrics["directional_error"] = LateralLongitudinalError(ppm)
    metrics["xy_seq_error"] = Location2DError("uv_seq", ppm)
    metrics["yaw_seq_error"] = AngleError("yaw_seq")
    metrics["directional_seq_error"] = LateralLongitudinalError(ppm, key="uv_seq")
    if has_gps:
        metrics["xy_gps_error"] = Location2DError("uv_gps", ppm)
        metrics["xy_gps_seq_error"] = Location2DError("uv_gps_seq", ppm)
        metrics["yaw_gps_seq_error"] = AngleError("yaw_gps_seq")
    metrics = metrics.to(module.device)

    keys_save = ["uvr_max", "uv_max", "yaw_max", "uv_expectation"]
    if has_gps:
        keys_save.append("uv_gps")
    if viz:
        keys_save.append("log_probs")

    for chunk_index, key in enumerate(tqdm(chunk_keys, disable=not progress)):
        indices = chunk2idx[key]
        aligner = RigidAligner(track_priors=viz, num_rotations=num_rotations)
        if has_gps:
            aligner_gps = GPSAligner(track_priors=viz, num_rotations=num_rotations)

        all_data = [dataset[i] for i in indices]
        if not all_data:
            continue

        collated_data = collate(all_data)
        collated_data = module.transfer_batch_to_device(
            collated_data, module.device, 0
        )
        pred_batch = model(collated_data)
        preds_list = unbatch_to_device(pred_batch)

        batches = []
        preds = []
        for i, pred in enumerate(preds_list):
            data = all_data[i]
            data = module.transfer_batch_to_device(data, module.device, 0)

            canvas = data["canvas"]
            data["xy_geo"] = xy = canvas.to_xy(data["uv"].double())
            data["yaw"] = yaw = data["roll_pitch_yaw"][-1].double()
            aligner.update(pred["log_probs"], canvas, xy, yaw)

            if has_gps:
                uv_gps = data["uv_gps"]
                pred["uv_gps"] = uv_gps
                xy_gps = canvas.to_xy(uv_gps.double())
                aligner_gps.update(xy_gps, data["accuracy_gps"], canvas, xy, yaw)

            if not viz:
                data.pop("image", None)
                data.pop("map", None)
            batches.append(data)
            preds.append({k: pred[k] for k in keys_save if k in pred})

        xy_gt = torch.stack([b["xy_geo"] for b in batches])
        yaw_gt = torch.stack([b["yaw"] for b in batches])
        aligner.compute()
        xy_seq, yaw_seq = aligner.transform(xy_gt, yaw_gt)
        if has_gps:
            aligner_gps.compute()
            xy_gps_seq, yaw_gps_seq = aligner_gps.transform(xy_gt, yaw_gt)
        results = []
        for i in range(len(indices)):
            preds[i]["uv_seq"] = batches[i]["canvas"].to_uv(xy_seq[i]).float()
            preds[i]["yaw_seq"] = yaw_seq[i].float()
            if has_gps:
                preds[i]["uv_gps_seq"] = (
                    batches[i]["canvas"].to_uv(xy_gps_seq[i]).float()
                )
                preds[i]["yaw_gps_seq"] = yaw_gps_seq[i].float()
            results.append(metrics(preds[i], batches[i]))
        if viz:
            callback(chunk_index, module, batches, preds, results, aligner)
        del aligner, preds, batches, results, pred_batch, preds_list, all_data
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
    gpus: int = 1,
    **kwargs,
):
    if experiment in pretrained_models:
        experiment, cfg_override = pretrained_models[experiment]
        cfg = OmegaConf.merge(OmegaConf.create(dict(model=cfg_override)), cfg)

    is_distributed = gpus > 1
    rank = dist.get_rank() if is_distributed else 0
    if rank == 0:
        logger.info("Evaluating model %s with config %s", experiment, cfg)
    
    checkpoint_path = resolve_checkpoint_path(experiment)
    model = GenericModule.load_from_checkpoint(
        checkpoint_path, cfg=cfg, find_best=not experiment.endswith(".ckpt")
    )
    model = model.eval()

    device = 'cpu'
    if torch.cuda.is_available():
        if is_distributed:
            device = torch.device(f'cuda:{rank}')
            model = model.to(device)
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        else:
            device = torch.device('cuda')
            model = model.to(device)

    if is_distributed:
        dist.barrier()
    dataset.prepare_data()
    if is_distributed:
        dist.barrier()
    dataset.setup()
    if is_distributed:
        dist.barrier()

    if output_dir is not None and rank == 0:
        output_dir.mkdir(exist_ok=True, parents=True)
        if callback is None:
            if sequential:
                callback = plot_example_sequential
            else:
                callback = plot_example_single
            callback = functools.partial(
                callback, out_dir=output_dir, **(viz_kwargs or {})
            )
    kwargs = {**kwargs, "callback": callback if rank == 0 else None, "gpus": gpus}

    seed_everything(dataset.cfg.seed + rank)
    if sequential:
        dset, chunk2idx = dataset.sequence_dataset(split, **cfg.chunking)
        metrics = evaluate_sequential(dset, chunk2idx, model, **kwargs)
    else:
        loader = dataset.dataloader(split, shuffle=False, num_workers=num_workers)
        metrics = evaluate_single_image(loader, model, **kwargs)

    if is_distributed:
        dist.barrier()

    if rank == 0:
        results = metrics.compute()
        logger.info("All results: %s", {k: v.item() if torch.is_tensor(v) and v.numel() == 1 else v for k, v in results.items()})
        if output_dir is not None:
            write_dump(output_dir, experiment, cfg, results, metrics)
            logger.info("Outputs have been written to %s.", output_dir)
    
    return metrics