# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
from pathlib import Path
from typing import Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from .. import logger
from ..conf import data as conf_data_dir
from ..data import MapillaryDataModule
from .run import evaluate

# 可评估的场景;每个对应 conf/data/<scene>.yaml 与 splits_RHO_<scene>.json。
# CV-FSS(datasets/CV-FSS/)+ CV-RHO(datasets/CV-RHO/,场景 yaml 内 data_dir 指向 CV-RHO)。
SCENES = [
    "neuanspach", "tirol", "bad_herrenalb", "speicher", "overgaard",                        # CV-FSS
    "chicago", "detroit", "sanfrancisco", "washington", "montrouge", "toulouse", "berlin",  # CV-RHO
]

CONF_DATA_DIR = Path(conf_data_dir.__file__).parent

# 评估期对共享场景 data 配置(conf/data/<scene>.yaml)的固定覆盖:场景 yaml 只含
# 共享部分,这里补 onet 专属的 datamodule 名(mapillary,非 pano)、keep_view=view1(单帧+
# 序列都只用 view1 这一条 stream,与 RHO 每条轨迹整条对称、只是单视图)、road prior(序列
# 聚合 ε-recovery 硬依赖 data["road_prior"]),再叠加评估用的小初始误差与 val 小 batch。
EVAL_DATA_OVERRIDE = {
    "name": "mapillary",
    "keep_view": "view1",
    "add_road_prior": True,
    "road_prior_sigma": 10.0,
    "add_map_mask": True,
    "max_init_error": 32,
    "loading": {"val": {"batch_size": 4, "num_workers": 4}},
}

def build_cfg(scene: str) -> DictConfig:
    """基于共享场景 data 配置(conf/data/<scene>.yaml)构建评估配置。场景 yaml 只含
    共享部分,EVAL_DATA_OVERRIDE 补 onet 专属的 datamodule 名(mapillary)、view1 过滤、
    road prior 及评估期小 batch/初始误差。序列由 datamodule 自排序整条(不 chunk、对齐
    RHO),故无需 chunking 配置。模型沿用 checkpoint(orienternet 原版),不额外指定。"""
    data_cfg = OmegaConf.load(CONF_DATA_DIR / f"{scene}.yaml")
    data_cfg = OmegaConf.merge(data_cfg, EVAL_DATA_OVERRIDE)
    return OmegaConf.create({"data": data_cfg})


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
    dataset = MapillaryDataModule(cfg.get("data", {}))

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
        help="评估场景;对应 conf/data/<scene>.yaml 与 splits_RHO_<scene>.json",
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
