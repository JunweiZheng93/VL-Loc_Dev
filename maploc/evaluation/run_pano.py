# Copyright (c) Meta Platforms, Inc. and affiliates.

import functools
from itertools import islice
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from torchmetrics import MetricCollection
from tqdm import tqdm

from .. import EXPERIMENTS_PATH, logger
from ..data.torch import collate, unbatch_to_device
from ..models.metrics_v1gt import AngleError, LateralLongitudinalError, Location2DError
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
    """Single-frame evaluation, one panorama (3 ordered views) per batch.

    The val loader (batch_size=3, no shuffle, worker processes) yields the
    packed views in panorama order, so every batch is one view1/view2/view3
    triple and the workers overlap the preprocessing with the GPU forward.
    `num` limits the number of batches.
    """
    num_views = 3
    ppm = model.model.conf.pixel_per_meter
    metrics = MetricCollection(model.model.metrics())
    metrics["directional_error"] = LateralLongitudinalError(ppm)
    metrics = metrics.to(model.device)

    for i, batch_ in enumerate(
        islice(tqdm(dataloader, total=num, disable=not progress), num)
    ):
        # The model treats every 3 consecutive samples as a panorama; guard
        # against silently pairing views of different panoramas (or out of order).
        names = batch_["name"]
        for v in range(0, len(names), num_views):
            triple = names[v : v + num_views]
            assert len({n.rsplit("_", 1)[0] for n in triple}) == 1 and [
                n.rsplit("_", 1)[1] for n in triple
            ] == ["view1", "view2", "view3"], f"Not a valid panorama triple: {triple}"
        batch = model.transfer_batch_to_device(batch_, model.device, i)

        # Ablation: mask semantic classes on all views
        if mask_index is not None:
            for view_idx in range(batch["map"].shape[0]):
                mask = batch["map"][view_idx, mask_index[0]] == (mask_index[1] + 1)
                batch["map"][view_idx, mask_index[0]][mask] = 0

        pred = model(batch)

        # The metrics handle the panorama logic via the 'is_panorama' flag
        results = metrics(pred, batch)
        if callback is not None:
            # Visualization shows the main view (view1) of the first panorama
            gt_data = {k: v[0] for k, v in batch_.items()}
            callback(
                i, model, unbatch_to_device(pred), unbatch_to_device(gt_data), results
            )
        del batch_, batch, pred, results

    return metrics.cpu()


@torch.no_grad()
def evaluate_sequential(
    loader: torch.utils.data.DataLoader,
    seq_keys: List,
    seq2idx: Dict,
    model: GenericModule,
    num: Optional[int] = None,
    callback: Optional[Callable] = None,
    progress: bool = True,
    mask_index: Optional[Tuple[int]] = None,
    recovery_eps: float = 0.01,
    obs_weight_gamma: float = 1.0,
    readout_radius: int = 3,
):
    """Sequence evaluation over `datamodule.sequence_dataloader(split)`.

    The loader yields the single samples trajectory by trajectory in the order
    of `seq_keys` (view1/view2/view3 per panorama), preprocessed in worker
    processes; shuffling is decided when the loader is built. `num` limits the
    number of trajectories.
    """
    num_views = 3
    if num is not None:
        seq_keys = seq_keys[:num]
    lengths = [len(seq2idx[k]) // num_views for k in seq_keys] # Length in panoramas
    logger.info(
        "Min/med/max trajectory lengths: %d/%d/%d, total number of panoramas: %d",
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

    samples = iter(loader)
    for seq_index, key in enumerate(tqdm(seq_keys, disable=not progress)):
        num_panoramas = len(seq2idx[key]) // num_views
        aligner = RigidAligner()

        gt_batches = []
        preds = []
        uvt_online = []

        for i in range(num_panoramas):
            # The next 3 samples of the loader are the views of one panorama
            batch_list = [next(samples) for _ in range(num_views)]
            # The model treats any batch of 3 as a panorama; guard against
            # silently pairing views of different panoramas (or out of order).
            names = [item["name"] for item in batch_list]
            assert len({n.rsplit("_", 1)[0] for n in names}) == 1 and [
                n.rsplit("_", 1)[1] for n in names
            ] == ["view1", "view2", "view3"], f"Not a valid panorama triple: {names}"
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
            aligner.update_with_ref(pred["log_probs"][0], canvas, xy, yaw, recovery_eps=recovery_eps, obs_weight_gamma=obs_weight_gamma, recovery_prior=gt_data["road_prior"])
            uvt_online.append(local_expectation_xyr(aligner.belief, readout_radius))

            if not viz:
                # To save memory, we can pop large tensors if not visualizing
                for item in batch_list:
                    item.pop("image", None)
                    item.pop("map", None)
            
            gt_batches.append(gt_data)
            preds.append({k: pred[k][0] for k in keys_save})
            del pred, batch, batch_

        results = []
        for i in range(num_panoramas):
            preds[i]["uv_seq"] = uvt_online[i][:2].float()
            preds[i]["yaw_seq"] = uvt_online[i][-1].float()

            # For metrics, we need to pass the full batch (for all 3 views)
            # but our metrics are already adapted to handle this via 'is_panorama'
            # and will internally select the correct GT.
            # Here we can just pass the GT of the main view for simplicity, as
            # the metrics will only use the keys from this dict.
            results.append(metrics(preds[i], gt_batches[i]))
            
        if viz:
            callback(seq_index, model, gt_batches, preds, results, aligner)
        del aligner, uvt_online, preds, gt_batches, results

    return metrics.cpu()


def evaluate(
    experiment: str,
    cfg: DictConfig,
    dataset,
    split: str,
    sequential: bool = False,
    output_dir: Optional[Path] = None,
    callback: Optional[Callable] = None,
    num_workers: Optional[int] = None,  # None: num_workers of the data config
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
    # 预处理交给 DataLoader worker(worker 内 torch 线程数为 1),但主进程仍有少量 CPU 张量
    # 运算,而主进程的 intra-op 线程数默认等于机器核数:共享机器上只要有一个核被别的进程占着,
    # 每个 OpenMP 并行区都会在 barrier 上等调度(实测慢 15 倍以上),故限制线程数。
    torch.set_num_threads(8)
    if sequential:
        loader, seq_keys, seq2idx = dataset.sequence_dataloader(
            split, shuffle=kwargs.pop("shuffle", False)
        )
        metrics = evaluate_sequential(loader, seq_keys, seq2idx, model, **kwargs)
    else:
        # val loader: batch_size=3 = one ordered panorama per batch, no shuffle,
        # preprocessing in worker processes overlapped with the GPU forward.
        loader = dataset.dataloader(split, shuffle=False, num_workers=num_workers)
        metrics = evaluate_single_image(loader, model, **kwargs)

    results = metrics.compute()
    logger.info("All results: %s", results)
    if output_dir is not None:
        write_dump(output_dir, experiment, cfg, results, metrics)
        logger.info("Outputs have been written to %s.", output_dir)
    return metrics