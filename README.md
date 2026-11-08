# VL-Loc

Visual localization from a street-level panorama, **text** and OpenStreetMap: the image
proposes pose hypotheses, language verifies them. The repository holds

- the two image-only baselines the project builds on: **RHO** (`orienternet_fused_prior`:
  OrienterNet-style localization of a three-view panorama with a fused xy / yaw prior and
  sequence-mode belief aggregation) and the original **OrienterNet** (`orienternet`);
- the text models **TextVerifyItems** (`orienternet_verify_items`, the current design) and its
  predecessor **TextVerify** (`orienternet_verify`);
- the **stage-A caption** generator (`maploc/data/stage_a_captions.py`), which turns the OSM
  facts visible from a view into a fixed-grammar description.

Scenes: the rural CV-FSS set (`neuanspach`, `tirol`, `bad_herrenalb`, `speicher`, `overgaard`)
and the urban CV-RHO set (`chicago`, `detroit`, `sanfrancisco`, `washington`, `montrouge`,
`toulouse`, `berlin`), each with single-frame and sequence evaluation.

## Layout

| path | content |
|---|---|
| `maploc/models/orienternet_fused_prior.py` | RHO image model: BEV projection, panorama stitching, exhaustive matching, fused xy / yaw prior |
| `maploc/models/orienternet_verify_items.py` | TextVerifyItems: item verifier over the three view captions |
| `maploc/models/orienternet_verify.py` | TextVerify: area-fraction verifier over the view1 caption (predecessor; base class of the above) |
| `maploc/models/text_verifier.py` | shared verifier parts: frozen T5 encoder, region-query cross-attention head, NMS of the posterior, listwise target, basin re-weighting |
| `maploc/models/egocentric.py` | OSM labels seen from a pose, camera-frame grids, FOV wedge (coordinate conventions) |
| `maploc/data/stage_a_captions.py` | stage-A captions: visible OSM facts per bearing sector × range band, rendering / parsing, item tensors; generation CLI |
| `maploc/data/dataset.py`, `maploc/data/mapillary/dataset.py` | datasets and datamodules (captions, uncut `map_desc` raster, panorama-consistent flip) |
| `maploc/conf/{rho,verify,verify_items,onet}.yaml` | training top-levels; `maploc/conf/data/<scene>.yaml` per scene |
| `maploc/evaluation/eval_rho.py`, `maploc/evaluation/eval_onet.py` | evaluation entry points |

## Setup

```bash
pip install -r requirements/full.txt
```

Tested with torch 2.15.0 (cu132) / pytorch-lightning 2.6.6 / transformers 5.22.2. The text
models need `transformers`, `sentencepiece` and `protobuf`; the frozen `t5-base` encoder is
downloaded from the Hugging Face hub on first use. Before moving to another machine, run
`nvidia-smi` to check the driver's supported CUDA version and install a matching torch build.

## Data

Each scene lives under `datasets/CV-FSS/<scene>/` or `datasets/CV-RHO/<scene>/` with
`dump.json`, `tiles.pkl` and `images/`, next to the split file `splits_RHO_<scene>.json`.
The per-scene config `maploc/conf/data/<scene>.yaml` names the dataset root (`data_dir`); it is
the only difference between the two datasets, everything else is shared. Run all commands from
the repository root (hydra `chdir: false`) so that the relative `data_dir` resolves.

**Captions** are flat json dicts `{"<pano_id>_view{1,2,3}.jpg": "<caption>"}` under
`datasets/<dataset>/captions/`:

- `<scene>_stageA.json` (+ `<scene>_stageA_facts.pt`): **stage-A captions**, synthesised from the
  map at the ground-truth pose of every view. Loaded with `data.captions=stageA`. Per view, the 120°
  wedge up to 64 m is split into 5 bearing sectors × 3 range bands (0–10 / 10–30 / 30–64 m);
  visibility is ray-cast through buildings and forest; each visible region states the coverage
  level of its area classes, its ways (road, path, ...) and up to three point features, rendered
  into a fixed grammar (`"10 to 30 m away: mostly grass, a road ahead; buildings on the right."`)
  that `parse_caption` inverts exactly. Stage A is the first half of the caption pipeline; the
  rewriting of these facts into natural descriptions by a multimodal LLM (stage B) is done outside
  this repository. Generate them with

  ```bash
  python -m maploc.data.stage_a_captions --scene <scene>   # writes captions/<scene>_stageA.json and _facts.pt
  ```

- `<scene>.json`: free-form captions of the view images from a vision-language model (produced
  outside this repository). Loaded with `data.captions=true`.

Every sample then carries `caption` and `flip` (whether the training augmentation mirrored it).
`data.desc_crop_meters=<m>` adds `map_desc`, a second, uncut OSM raster of half-width `m` around the
same centre, on which the item verifier computes its descriptors (the ±64 m model crop cuts off the
64 m wedge of off-centre poses).

## RHO baseline

The `rho` top-level fixes the `orienternet_fused_prior` model and single-GPU training from
scratch: micro-batch 6 × accumulate 6 = effective batch 36, lr 1e-5, 30 epochs, validation every
500 micro-batches. The micro-batch must be a multiple of 3 (one panorama = 3 views; the forward
switches to panorama fusion when the batch size is a multiple of 3).

```bash
CUDA_VISIBLE_DEVICES=0 python -m maploc.train --config-name=rho \
  data=<scene> experiment.name=RHO_<scene> model.pano_geometry_fix=true
```

Checkpoints go to `experiments/RHO_<scene>/`: the three best epochs by `loss/total/val`
(`checkpoint-epoch=XX.ckpt`) and `last.ckpt`.

**Panorama geometry and augmentation.** Two defects of the inherited RHO code were fixed:

- *Training flip* (always on): the three views of a panorama now share one flip decision, a mirrored
  panorama is stitched with its side views swapped (a mirror swaps left and right of the rig), and the
  camera roll is negated for the mirrored image before rectification.
- *Panorama template geometry* (`model.pano_geometry_fix`, default `false`): the panorama template
  used to sample the 4H × W stitched BEV as if it were a 129 × 129 grid (squeezed by 1.71 × 1.98),
  rotated the side views about a point 0.5 m in front of the camera and normalised the panorama
  scores with the single-view template's cell count. `true` fixes all three. The flag changes the
  forward pass, so evaluate every checkpoint with the value it was trained with (it is stored in the
  run's config; configs without the key are legacy). New runs should set it to `true`.

## Evaluation

```bash
python -m maploc.evaluation.eval_rho --experiment RHO_<scene> --scene <scene> \
  model.num_rotations=256 [--sequential] [--output_dir eval_logs/<name>]
```

- `--experiment` takes a run name (`RHO_<scene>`: `last.ckpt` is read and the checkpoint with the
  lowest monitored loss is loaded) or a checkpoint path (`RHO_<scene>/checkpoint-epoch=10.ckpt`).
- `--scene` selects `conf/data/<scene>.yaml` and `splits_RHO_<scene>.json`.
- The model defaults to `orienternet_fused_prior`; the road prior applies only in the sequence
  aggregation, not in the single-frame forward. Other models need the dotlist override
  `model.name=<name>` (see the text models below).
- `--sequential` evaluates full trajectories with the recursive filter (`maploc/models/sequential.py`);
  knobs `--recovery_eps 0.01 --obs_weight_gamma 1.0 --readout_radius 3` (road-prior recovery,
  entropy-weighted observations, local-expectation readout). Without it, single frames.
- `--output_dir` writes `log.json` with the summary and the per-sample errors.

It prints the xy / yaw errors and `Recall @1/3/5` (m / °); in sequence mode also `xy_seq_error` /
`yaw_seq_error`. The sequence length is the full trajectory from the split file.

## Text models

Both text models keep the RHO image branch and training protocol unchanged (from scratch, only T5
frozen, the verifier parameters at `training.lr_text`) and let language **choose between the
image's pose hypotheses**: the image posterior is reduced to its top-K modes by non-maximum
suppression (5 m / 20°), a verifier scores every mode from the caption(s), and the posterior is
re-weighted basin by basin (the image shape is kept inside a basin, the verifier ranks the basins),
so `uv_max` is the selected mode and the sequence filter consumes the re-weighted `log_probs`.
The image NLL is computed on the pure image posterior, so the image branch trains exactly as the
baseline and `uv_max_img` / `xy_max_error_img` is a clean baseline read-out of the same forward pass.
The LR schedule and the checkpoint selection monitor `loss/nll/val` (the image NLL).

Extra validation metrics: `xy_max_error_img` / `yaw_max_error_img` (image arg-max), `select_acc`
(the selected mode is the one nearest to the ground truth, over samples that have a mode within
5 m / 20°) and `mode_recall` (such a mode exists among the K).

### TextVerifyItems (`orienternet_verify_items`, `conf/verify_items.yaml`)

For each of the three view captions a region-query head (frozen `t5-base` tokens → 15 region
queries with cross-attention) predicts the stage-A items of that view: the coverage level of every
area class, the ways and the point features of the 5 × 3 regions. Every mode is scored by the soft
Jaccard between the predicted items and the items the stage-A code computes at that mode on the
uncut `map_desc` raster (views 2 / 3 at ±120° from view 1); the scores enter a softmax over the
modes with a learned temperature and an optional image log-prob term. Losses: image NLL + item
cross-entropy (every sample) + listwise cross-entropy over the modes (samples without a mode near the
ground truth contribute 0).

```bash
# train (stage-A captions); add model.verify.blank_captions=true for the no-text control,
# or data.captions=true to train on the VLM captions
python -m maploc.train --config-name=verify_items data=<scene> \
  experiment.name=TextVerifyItems_RHO_<scene>_stageA model.pano_geometry_fix=true

# evaluate
python -m maploc.evaluation.eval_rho --experiment TextVerifyItems_RHO_<scene>_stageA --scene <scene> \
  model.name=orienternet_verify_items model.num_rotations=256 data.captions=stageA data.desc_crop_meters=136 \
  [--sequential] [--output_dir eval_logs/<name>]
```

The `model.name`, `data.captions` and `data.desc_crop_meters` overrides are required at evaluation:
`eval_rho` fixes the model to `orienternet_fused_prior` and builds the datamodule before the
checkpoint config is merged. `model.verify.blank_captions=true` at evaluation blanks the captions of
a trained model (test-time control); `model.verify.K_eval=20` scores more hypotheses without
retraining.

### TextVerify (`orienternet_verify`, `conf/verify.yaml`)

The first verifier: the `view1` caption predicts the OSM area-class fractions of 5 bearing
sectors × 3 range bins (0–32 m, 2 m grid) and every mode is scored by the L1 similarity of its
fractions. It is kept as an ablation and because TextVerifyItems inherits from it.

```bash
python -m maploc.train --config-name=verify data=<scene> experiment.name=TextVerify_RHO_<scene>
python -m maploc.evaluation.eval_rho --experiment TextVerify_RHO_<scene> --scene <scene> \
  model.name=orienternet_verify data.captions=true model.num_rotations=256 [--sequential]
```

## OrienterNet baseline (onet)

The OrienterNet baseline mirrors the RHO line on the same per-scene configs: the original
`orienternet` model (`x_max=19`), the non-panorama `mapillary` datamodule, no road prior, effective
batch 16 (micro-batch 8 × accumulate 2).

```bash
python -m maploc.train --config-name=onet data=<scene> experiment.name=Onet_<scene>
python -m maploc.evaluation.eval_onet --experiment Onet_<scene> --sequential --scene <scene> model.num_rotations=256
```

Its sequence aggregation uses the same recursive filter as RHO (`--recovery_eps`,
`--obs_weight_gamma`, `--readout_radius` behave identically; the eval config enables the road prior
for the recovery term). Evaluation uses only the `view1` stream (`keep_view`), so single-frame and
sequence counts match the RHO line one-to-one: one view per step instead of a three-view panorama.
Sequences are the full `view1` stream ordered by capture time, without chunking.

## Development notes

- Two earlier text mechanisms were implemented, measured and removed on 2026-11-07: LR-BEV (the
  caption rendered into an egocentric OSM sketch and matched by class agreement) and a probe that
  predicted the same sketch from image BEV features. Their write-ups live in `docs/`
  (`research-LR-BEV-*.md`), their checkpoints under `experiments/Text_RHO_*`.
- `docs/` (research notes and design specs) and `maploc/dev/` (tests of the models, of the panorama
  geometry fixes and of the stage-A code; preliminary studies such as top-K selection ceilings and
  caption / image → descriptor regressions; paired analyses of evaluation dumps) are not tracked by git.
- `experiments/` and `eval_logs/` may be symlinks to a results volume.
