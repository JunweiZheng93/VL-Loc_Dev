# RHO — Sequence-Mode Map Localization

OrienterNet-style visual map localization for feature-sparse rural scenes
(the CV-FSS dataset), with single-frame and sequence (online belief
aggregation) evaluation. Five scenes: `neuanspach`, `tirol`, `bad_herrenalb`,
`speicher`, `overgaard`.

## Setup

```bash
pip install -r requirements/full.txt
```

Tested with torch 2.12.1 (cu130) / pytorch-lightning 2.6.5 / CUDA 13.0. Before
moving to another machine, run `nvidia-smi` to check the driver's supported CUDA
version; if the driver is older, install a matching torch build instead of cu130.

## Data

Place each scene under `datasets/CV-FSS/<scene>/` containing `dump.json`,
`tiles.pkl`, and `images/`, plus the split file
`datasets/CV-FSS/splits_RHO_<scene>.json`.

A second dataset **CV-RHO** (identical layout, cities `chicago`, `detroit`,
`sanfrancisco`, `washington`, `montrouge`, `toulouse`, `berlin`) lives under
`datasets/CV-RHO/`. Every per-scene config `conf/data/seqloc_<scene>.yaml`
declares its `data_dir` explicitly (`datasets/CV-FSS` or `datasets/CV-RHO`) —
that value is the only difference between the two datasets' configs; the
training top-levels, eval scripts and `--scene` handling are all shared. Run
everything from the repo root (hydra `chdir: false`), so the relative
`data_dir` resolves.

## Training (single GPU)

The `seqloc_rho` top-level config fixes the `fused_prior` model and the
single-GPU training parameters. It trains from scratch and is otherwise
equivalent to the reference `RHO_neuanspach` run (same model / lr / schedule /
augmentation / effective batch); only single-GPU and from-scratch differ.

```bash
CUDA_VISIBLE_DEVICES=0 python -m maploc.train --config-name=seqloc_rho \
  data=seqloc_<scene> experiment.name=RHO_<scene>
```

For example `data=seqloc_neuanspach experiment.name=RHO_neuanspach`. Checkpoints go
to `experiments/RHO_<scene>/` (`best.ckpt` = best val loss). The effective batch
is 36 (micro-batch 6 × accumulate 6; micro-batch 12 OOMs on a 40GB A100, and the
micro-batch must be a multiple of 3 — one panorama = 3 views). To switch scenes,
change only `data=seqloc_<scene>` and `experiment.name=RHO_<scene>`. The per-scene
configs `conf/data/seqloc_<scene>.yaml` are shared by both the RHO and OrienterNet
lines; the datamodule (`mapillary_pano` vs `mapillary`) and road prior are set by
the top-level config, not the scene file.

## Sequence evaluation

```bash
python -m maploc.evaluation.seqloc_rho \
  --experiment RHO_<scene>/best.ckpt --sequential --scene <scene> \
  model.num_rotations=256
```

- `--scene` selects the scene (auto-maps to `conf/data/seqloc_<scene>.yaml` and
  `splits_RHO_<scene>.json`) — no need to hand-edit the script or config.
- The model is fixed to `orienternet_fused_prior`: the road prior applies only
  in the sequence-aggregation layer, not in the single-frame forward. (To try a
  different model, append the dotlist override `model.name=<name>`.)
- Drop `--sequential` for single-frame evaluation.
- Sequence-mode knobs (defaults shown): `--recovery_eps 0.01
  --obs_weight_gamma 1.0 --readout_radius 3` — road-prior recovery + entropy
  weighting + iso local-expectation readout.

It prints `xy_seq_error` / `yaw_seq_error` and `Recall @1/3/5`. The sequence
length is the full trajectory (in RHO one panorama = 3 pinhole views);
trajectories come from the split file.

## OrienterNet baseline (onet)

The OrienterNet baseline mirrors the RHO line and shares the same per-scene
configs. It uses the original `orienternet` model (`x_max=19`), the non-panorama
`mapillary` datamodule, no road prior, and a single-GPU effective batch of 16
(micro-batch 8 × accumulate 2), matching the reference run.

```bash
# train
python -m maploc.train --config-name=seqloc_onet data=seqloc_<scene> experiment.name=Onet_<scene>
# evaluate
python -m maploc.evaluation.seqloc_onet --experiment Onet_<scene>/best.ckpt \
  --sequential --scene <scene> model.num_rotations=256
```

Onet's sequence aggregation now uses the **same** recursive-filter aligner as the
RHO line (`maploc/models/sequential.py`), so the three contributions —
entropy-weighted observations, ε-recovery with the road prior, and the iso
local-expectation readout — are evaluated on the Onet backbone as well. The knobs
`--recovery_eps / --obs_weight_gamma / --readout_radius` behave identically to the
RHO line, and Onet's eval config also enables `add_road_prior` (ε-recovery reads
`data["road_prior"]`). The old markov aligner has been removed. Onet evaluation
uses only the `view1` stream (`keep_view`), so it mirrors RHO one-to-one: on
neuanspach val, 17 full-trajectory sequences and 1663 single-frame images — the
same counts as RHO — differing only in one view per step vs. a 3-view panorama.
Sequences are the full view1 stream ordered by capture time, with no chunking
(`MapillaryDataModule.sequence_dataset` self-sorts like the RHO datamodule, so
there is no `max_length` knob). Both single-frame and sequence recall are reported;
drop `--sequential` for single-frame only. Split naming: `splits_MODEL_CITY.json`.
