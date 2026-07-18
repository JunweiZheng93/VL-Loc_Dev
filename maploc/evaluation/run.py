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
from ..models.metrics import AngleError, LateralLongitudinalError, Location2DError
from ..models.sequential import RigidAligner, local_expectation_xyr
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
):
    ppm = model.model.conf.pixel_per_meter
    metrics = MetricCollection(model.model.metrics())
    metrics["directional_error"] = LateralLongitudinalError(ppm)
    metrics = metrics.to(model.device)

    for i, batch_ in enumerate(
        islice(tqdm(dataloader, total=num, disable=not progress), num)
    ):
        batch = model.transfer_batch_to_device(batch_, model.device, i)
        # Ablation: mask semantic classes
        if mask_index is not None:
            mask = batch["map"][0, mask_index[0]] == (mask_index[1] + 1)
            batch["map"][0, mask_index[0]][mask] = 0
        pred = model(batch)

        results = metrics(pred, batch)
        if callback is not None:
            callback(
                i, model, unbatch_to_device(pred), unbatch_to_device(batch_), results
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
    mask_index: Optional[Tuple[int]] = None,
    recovery_eps: float = 0.01,
    obs_weight_gamma: float = 1.0,
    readout_radius: int = 3,
):
    seq_keys = list(seq2idx)
    if shuffle:
        seq_keys = [seq_keys[i] for i in torch.randperm(len(seq_keys))]
    if num is not None:
        seq_keys = seq_keys[:num]
    lengths = [len(seq2idx[k]) for k in seq_keys]
    logger.info(
        "Min/max/med lengths: %d/%d/%d, total number of images: %d",
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
    metrics = metrics.to(model.device)

    keys_save = ["uvr_max", "uv_max", "yaw_max", "uv_expectation"]
    if viz:
        keys_save.append("log_probs")

    for seq_index, key in enumerate(tqdm(seq_keys, disable=not progress)):
        indices = seq2idx[key]
        aligner = RigidAligner()
        batches = []
        preds = []
        uvt_online = []
        for i in indices:
            data = dataset[i]
            data = model.transfer_batch_to_device(data, model.device, 0)
            pred = model(collate([data]))

            canvas = data["canvas"]
            data["xy_geo"] = xy = canvas.to_xy(data["uv"].double())
            data["yaw"] = yaw = data["roll_pitch_yaw"][-1].double()
            aligner.update_with_ref(
                pred["log_probs"][0],
                canvas,
                xy,
                yaw,
                recovery_eps=recovery_eps,
                obs_weight_gamma=obs_weight_gamma,
                recovery_prior=data["road_prior"],
            )
            uvt_online.append(local_expectation_xyr(aligner.belief, readout_radius))

            if not viz:
                data.pop("image")
                data.pop("map")
            batches.append(data)
            preds.append({k: pred[k][0] for k in keys_save})
            del pred

        results = []
        for i in range(len(indices)):
            preds[i]["uv_seq"] = uvt_online[i][:2].float()
            preds[i]["yaw_seq"] = uvt_online[i][-1].float()
            results.append(metrics(preds[i], batches[i]))
        if viz:
            callback(seq_index, model, batches, preds, results, aligner)
        del aligner, uvt_online, preds, batches, results
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
        checkpoint_path, cfg=cfg, find_best=not experiment.endswith(".ckpt"), strict=False
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
        loader = dataset.dataloader(split, shuffle=True, num_workers=num_workers)
        metrics = evaluate_single_image(loader, model, **kwargs)

    results = metrics.compute()
    logger.info("All results: %s", results)
    if output_dir is not None:
        write_dump(output_dir, experiment, cfg, results, metrics)
        logger.info("Outputs have been written to %s.", output_dir)
    return metrics
