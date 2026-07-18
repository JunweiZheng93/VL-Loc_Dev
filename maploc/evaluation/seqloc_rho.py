# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
from pathlib import Path
from typing import Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from .. import logger
from ..conf import data as conf_data_dir
from ..data import MapillaryPanoDataModule
from .run_pano import evaluate

# 可评估的场景;每个对应 conf/data/seqloc_<scene>.yaml 与 splits_RHO_<scene>.json。
# CV-FSS(datasets/CV-FSS/)+ CV-RHO(datasets/CV-RHO/,场景 yaml 内 data_dir 指向 CV-RHO)。
SCENES = [
    "neuanspach", "tirol", "bad_herrenalb", "speicher", "overgaard",                        # CV-FSS
    "chicago", "detroit", "sanfrancisco", "washington", "montrouge", "toulouse", "berlin",  # CV-RHO
]

# 评估固定用 fused_prior:road prior 只在序列聚合层生效,单帧 forward 不叠加,
# 与训练 forward 一致、归因干净。(真要临时换模型,命令行加 dotlist model.name=xxx 即可覆盖。)
MODEL_NAME = "orienternet_fused_prior"

CONF_DATA_DIR = Path(conf_data_dir.__file__).parent

# 评估期对共享场景 data 配置(conf/data/seqloc_<scene>.yaml)的固定覆盖:
# 场景 yaml 只含共享部分,这里补 RHO 专属的 datamodule 名与 road prior,
# 再叠加评估用的小初始误差与 val 小 batch。
EVAL_DATA_OVERRIDE = {
    "name": "mapillary_pano",
    "add_road_prior": True,
    "road_prior_sigma": 10.0,
    "add_map_mask": True,
    "max_init_error": 32,
    "loading": {"val": {"batch_size": 3, "num_workers": 3}},
}


def build_cfg(scene: str) -> DictConfig:
    """基于共享场景 data 配置(conf/data/seqloc_<scene>.yaml)构建评估配置。场景 yaml 只含
    共享部分,EVAL_DATA_OVERRIDE 补 RHO 专属的 datamodule 名与 road prior 及评估期
    小 batch/初始误差。模型固定 fused_prior。"""
    data_cfg = OmegaConf.load(CONF_DATA_DIR / f"seqloc_{scene}.yaml")
    data_cfg = OmegaConf.merge(data_cfg, EVAL_DATA_OVERRIDE)
    return OmegaConf.create({"data": data_cfg, "model": {"name": MODEL_NAME}})


def run(
    split: str,
    experiment: str,
    cfg: Optional[DictConfig] = None,
    sequential: bool = False,
    thresholds: Tuple[int] = (1, 3, 5),
    scene: str = "neuanspach",
    **kwargs,
):
    cfg = cfg or {}
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    default = build_cfg(scene)
    cfg = OmegaConf.merge(default, cfg)  # 命令行 dotlist 可在此基础上再覆盖
    dataset = MapillaryPanoDataModule(cfg.get("data", {}))

    metrics = evaluate(experiment, cfg, dataset, split, sequential=sequential, **kwargs)

    keys = [
        "xy_max_error",
        "yaw_max_error",
    ]
    if sequential:
        keys += [
            "xy_seq_error",
            "yaw_seq_error",
        ]
    for k in keys:
        if k not in metrics:
            logger.warning("Key %s not in metrics.", k)
            continue
        rec = metrics[k].recall(thresholds).double().numpy().round(2).tolist()
        logger.info("Recall %s: %s at %s m/°", k, rec, thresholds)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument(
        "--scene", type=str, default="neuanspach", choices=SCENES,
        help="评估场景;对应 conf/data/seqloc_<scene>.yaml 与 splits_RHO_<scene>.json",
    )
    parser.add_argument("--split", type=str, default="val", choices=["val"])
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--recovery_eps", type=float, default=0.01)
    parser.add_argument("--obs_weight_gamma", type=float, default=1.0, help="exponent of the entropy-based observation weight; 0 disables")
    parser.add_argument("--readout_radius", type=int, default=3, help="half-size of the local-expectation window for the sub-bin xy readout; 0 falls back to pure argmax")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--num", type=int)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_args()
    cfg = OmegaConf.from_cli(args.dotlist)
    kwargs = dict(output_dir=args.output_dir, num=args.num)
    if args.sequential:
        kwargs["recovery_eps"] = args.recovery_eps
        kwargs["obs_weight_gamma"] = args.obs_weight_gamma
        kwargs["readout_radius"] = args.readout_radius
    run(
        args.split,
        args.experiment,
        cfg,
        args.sequential,
        scene=args.scene,
        **kwargs,
    )
