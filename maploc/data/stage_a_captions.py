"""Stage-A reference captions: deterministic English descriptions of the OSM facts visible
from the GT pose of a view (spec docs/superpowers/specs/2026-11-07-stage-a-captions-design.md).

Camera frame as in maploc.models.egocentric: x right, z forward (metres), grid row 0 = far. Per view the
120 deg wedge up to 64 m is split into 5 bearing sectors x 3 range bands (0-10 / 10-30 / 30-64 m).
Visibility is ray-cast on the 0.5 m egocentric label grid: a ray starts to be blocked once it
enters a building or forest polygon from outside it (the camera's own polygon is transparent);
the entered polygon's cells along the ray stay visible (the object is seen), everything behind is not.
"""
import math
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F

from ..models.egocentric import egocentric_labels, fov_mask
from ..models.utils import make_grid
from ..osm.parser import Groups

AREAS = ["none"] + list(Groups.areas)  # 8 classes, index = raster value
WAYS = ["road", "path", "cycleway", "fence", "wall", "hedge"]
WAY_IDX = [Groups.ways.index(w) + 1 for w in WAYS]  # raster values of the ways channel
NODES = list(Groups.nodes)  # 33 classes, raster value = index + 1
SECTOR_PHRASES = ["far left", "on the left", "ahead", "on the right", "far right"]
ALL_ACROSS = "all across"
BAND_LABELS = ["Within 10 m", "10 to 30 m away", "30 to 64 m away"]
Z_MAX, X_MAX, CELL, FOV_DEG = 64.0, 56.0, 0.5, 120.0
SECTORS, RANGE_EDGES = 5, (10.0, 30.0, 64.0)
BANDS = len(RANGE_EDGES)
R = SECTORS * BANDS
OCCLUDERS = (1, 6)  # building, forest
RAY_STEP_DEG, RAY_STEP_M = 0.25, 0.25
VISIBLE_MIN = 0.10


class Geometry(NamedTuple):
    grid: torch.Tensor  # [h, w, 2] camera-frame cell centres (x, z), aligned with the 0.5 m raster pixels
    region: torch.Tensor  # [h, w] long: sector * 3 + band, -1 outside the wedge or beyond 64 m
    cell_count: torch.Tensor  # [R] long


def make_geometry() -> Geometry:
    # cell centres at multiples of CELL (the raster pitch; with an integer uv, as in the tests, nearest sampling
    # then hits pixel centres exactly - real uv are fractional and sampling stays plain nearest-neighbour)
    grid = make_grid(2 * X_MAX, Z_MAX, step_x=CELL, step_y=CELL, orig_x=-X_MAX, orig_y=CELL, y_up=True)
    x, z = grid[..., 0], grid[..., 1]
    r = torch.sqrt(x**2 + z**2)
    valid = fov_mask(grid, FOV_DEG, Z_MAX) & (r <= Z_MAX + 1e-6)
    half = math.radians(FOV_DEG / 2)
    sector = ((torch.atan2(x, z) + half) / (2 * half) * SECTORS).floor().clamp(0, SECTORS - 1).long()
    band = (r >= RANGE_EDGES[0]).long() + (r >= RANGE_EDGES[1]).long()
    region = torch.where(valid, sector * BANDS + band, torch.full_like(sector, -1))
    return Geometry(grid, region, torch.bincount(region[valid], minlength=R))


def cell_of(geo: Geometry, x: float, z: float):
    """(row, col) of the grid cell whose centre is nearest to the camera-frame point (x, z)."""
    h, w = geo.region.shape
    col = min(max(int(math.floor((x + X_MAX) / CELL + 0.5)), 0), w - 1)
    row = min(max(int(math.floor((Z_MAX - z) / CELL + 0.5)), 0), h - 1)
    return row, col


def _ray_samples(geo: Geometry) -> torch.Tensor:
    """Flat cell index of every ray sample: [n_rays, n_steps] (rays over the FOV, steps along range)."""
    h, w = geo.region.shape
    bearings = torch.deg2rad(torch.arange(-FOV_DEG / 2, FOV_DEG / 2 + 1e-6, RAY_STEP_DEG))
    ranges = torch.arange(RAY_STEP_M, Z_MAX + 1e-6, RAY_STEP_M)
    x = torch.sin(bearings)[:, None] * ranges[None]
    z = torch.cos(bearings)[:, None] * ranges[None]
    col = ((x + X_MAX) / CELL + 0.5).floor().clamp(0, w - 1).long()
    row = ((Z_MAX - z) / CELL + 0.5).floor().clamp(0, h - 1).long()
    return row * w + col


def ray_visibility(areas: torch.Tensor, geo: Geometry, cam_label: Optional[torch.Tensor] = None) -> torch.Tensor:
    """areas [B, h, w] long egocentric area labels -> [B, h, w] bool visible cells.
    cam_label [B]: area class of the camera cell (default: the grid cell at x = 0, z = CELL)."""
    B, h, w = areas.shape
    idx = _ray_samples(geo).to(areas.device)  # [n_rays, n_steps]
    lab = areas.reshape(B, -1)[:, idx]  # [B, n_rays, n_steps]
    if cam_label is None:
        cam_label = areas[:, h - 1, w // 2]
    prev = torch.cat([cam_label[:, None, None].expand(B, idx.shape[0], 1), lab[..., :-1]], -1)
    occ = (lab == OCCLUDERS[0]) | (lab == OCCLUDERS[1])
    hit = occ & (lab != prev)  # entering an occluder polygon from outside it
    after = hit.cumsum(-1) > 0
    first = hit.float().argmax(-1, keepdim=True)  # first hit (0 when none; harmless, masked by `after`)
    leave = after & (lab != lab.gather(-1, first))  # left the entered polygon: blocked from here on
    blocked = leave.cumsum(-1) > 0
    vis = torch.zeros(B, h * w, dtype=torch.bool, device=areas.device)
    b_idx = torch.arange(B, device=areas.device)[:, None, None].expand_as(lab)
    sel = ~blocked
    vis[b_idx[sel], idx[None].expand_as(lab)[sel]] = True
    return vis.view(B, h, w)


def _check_inside(uv: torch.Tensor, H: int, W: int, ppm: float):
    rad = Z_MAX * ppm  # the whole 64 m disc must lie inside the raster, whatever the yaw
    u, v = uv[..., 0], uv[..., 1]
    if not ((u >= rad) & (u <= W - 1 - rad) & (v >= rad) & (v <= H - 1 - rad)).all():
        raise ValueError("64 m wedge outside the map raster: enlarge the raster (crop / desc_crop_meters)")


def _facts_from_labels(labels: torch.Tensor, cam: torch.Tensor, geo: Geometry) -> dict:
    """labels [N,3,h,w] egocentric (areas, ways, dilated nodes), cam [N] area class of the camera cell."""
    dev = labels.device
    region = geo.region.to(dev)
    vis = ray_visibility(labels[:, 0], geo, cam) & (region >= 0)
    reg = F.one_hot(region.clamp(min=0), R).float() * (region >= 0)[..., None]  # [h, w, R]
    visf = vis.float()
    n_vis = torch.einsum("bhw,hwr->br", visf, reg)
    areas = F.one_hot(labels[:, 0].clamp(0, len(AREAS) - 1), len(AREAS)).float() * visf[..., None]
    areas = torch.einsum("bhwc,hwr->brc", areas, reg) / n_vis.clamp(min=1)[..., None]
    ways = F.one_hot(labels[:, 1].clamp(0, len(Groups.ways)), len(Groups.ways) + 1).float() * visf[..., None]
    ways = torch.einsum("bhwc,hwr->brc", ways, reg)[..., WAY_IDX].round().long()
    nodes1h = F.one_hot(labels[:, 2].clamp(0, len(NODES)), len(NODES) + 1).float() * visf[..., None]
    nodes_present = torch.einsum("bhwc,hwr->brc", nodes1h, reg)[..., 1:] > 0.5
    return {
        "areas": areas,
        "ways": ways,
        "nodes": nodes_present,
        "visible": n_vis / geo.cell_count.to(dev).float(),
        "camera_in_building": cam == 1,
        "camera_in_forest": cam == 6,
    }


def view_facts(map_r: torch.Tensor, uv: torch.Tensor, yaw: torch.Tensor, ppm: float, geo: Geometry) -> dict:
    """Visible OSM facts per region of every view. map_r [B,3,H,W] long raster, uv [B,2] px, yaw [B] deg.
    Returns areas [B,R,8] fractions among visible cells, ways [B,R,6] visible cell counts,
    nodes [B,R,33] presence, visible [B,R] visible fraction, camera_in_building / camera_in_forest [B]."""
    _check_inside(uv, *map_r.shape[-2:], ppm)
    dev = map_r.device
    nodes = F.max_pool2d(map_r[:, 2:3].float(), 3, 1, 1).long()  # 1-pixel nodes survive the nearest resampling
    labels = egocentric_labels(torch.cat([map_r[:, :2], nodes], 1), uv, yaw, geo.grid.to(dev), ppm)  # [B,3,h,w]
    cam = egocentric_labels(map_r, uv, yaw, torch.zeros(1, 1, 2, device=dev), ppm)[:, 0, 0, 0]
    return _facts_from_labels(labels, cam, geo)


def _sample_poses(raster: torch.Tensor, uv: torch.Tensor, yaw: torch.Tensor, grid_xz: torch.Tensor, ppm: float):
    """Nearest-neighbour samples of raster [B,C,H,W] on the camera-frame grid [h,w,2] seen from M poses per map
    (uv [B,M,2] px, yaw [B,M] deg), in one grid_sample call (same formulas as egocentric_labels): [B,M,C,h,w]."""
    B, C, H, W = raster.shape
    M = uv.shape[1]
    h, w = grid_xz.shape[:2]
    psi = torch.deg2rad(yaw.to(grid_xz))[..., None, None]  # [B,M,1,1]
    c, s = torch.cos(psi), torch.sin(psi)
    x, z = grid_xz[None, None, ..., 0], grid_xz[None, None, ..., 1]
    dx = c * x + s * z
    dy = -s * x + c * z
    u = uv[..., 0, None, None].to(grid_xz) + dx * ppm
    v = uv[..., 1, None, None].to(grid_xz) - dy * ppm
    grid = torch.stack([u / (W - 1) * 2 - 1, v / (H - 1) * 2 - 1], -1).reshape(B, M * h, w, 2)
    out = F.grid_sample(raster.float(), grid, mode="nearest", padding_mode="zeros", align_corners=True)
    return out.round().long().reshape(B, C, M, h, w).permute(0, 2, 1, 3, 4)


def view_facts_multi(map_r: torch.Tensor, uv: torch.Tensor, yaw: torch.Tensor, ppm: float, geo: Geometry) -> dict:
    """view_facts for M poses per map without copying the map: map_r [B,3,H,W], uv [B,M,2] px, yaw [B,M] deg.
    Every output gets the leading shape [B, M]."""
    _check_inside(uv, *map_r.shape[-2:], ppm)
    B, M = uv.shape[:2]
    dev = map_r.device
    nodes = F.max_pool2d(map_r[:, 2:3].float(), 3, 1, 1).long()
    labels = _sample_poses(torch.cat([map_r[:, :2], nodes], 1), uv, yaw, geo.grid.to(dev), ppm)
    cam = _sample_poses(map_r[:, :1], uv, yaw, torch.zeros(1, 1, 2, device=dev), ppm)[:, :, 0, 0, 0]
    f = _facts_from_labels(labels.reshape(B * M, *labels.shape[2:]), cam.reshape(B * M), geo)
    return {k: v.reshape(B, M, *v.shape[1:]) for k, v in f.items()}


# ---------------------------------------------------------------------------
# template (A3) and parse-back (A4)
# ---------------------------------------------------------------------------
AREA_NOUNS = {
    "none": "open ground", "building": "buildings", "parking": "parking", "playground": "playground",
    "grass": "grass", "park": "park", "forest": "forest", "water": "water",
}
LEVELS = ((0.60, "mostly"), (0.25, "about half"), (0.05, "some"))  # coverage among the visible cells
WAY_MIN_CELLS, MAX_NODES = 3, 3
EMPTY_CAPTION = "No mapped features are visible."
Item = Tuple[int, int, str]  # (band, sector, token)


def quantize_facts(facts: dict) -> dict:
    """Round areas / visible through float16 so that the captions are rendered from exactly the values the .pt keeps."""
    return {k: (v.half().float() if k in ("areas", "visible") else v) for k, v in facts.items()}


def _level(p: float) -> Optional[str]:
    for thr, word in LEVELS:
        if p >= thr:
            return word
    return None


_NOUN_CLASS = {noun: AREAS.index(cls) for cls, noun in AREA_NOUNS.items()}
_LEVEL_RANK = {word: i for i, (_, word) in enumerate(LEVELS)}


def _token_key(token: str):
    """Canonical order inside a region: areas (coverage level desc, then class index), ways, nodes."""
    for word, rank in _LEVEL_RANK.items():
        if token.startswith(word + " "):
            return (0, rank, _NOUN_CLASS[token[len(word) + 1:]])
    name = token[2:]  # "a <name>"
    if name in WAYS:
        return (1, WAYS.index(name))
    return (2, NODES.index(name.replace(" ", "_")))


def discretize(facts: dict, i: int) -> List[Item]:
    """Mention-level facts of view i, in rendering order: band, sector, areas (coverage desc), ways, nodes."""
    areas, ways = facts["areas"][i].tolist(), facts["ways"][i].tolist()
    nodes, visible = facts["nodes"][i].tolist(), facts["visible"][i].tolist()
    items: List[Item] = []
    for b in range(BANDS):
        for s in range(SECTORS):
            r = s * BANDS + b
            if visible[r] < VISIBLE_MIN:
                continue
            p = areas[r]
            top = max(range(len(AREAS)), key=lambda c: (p[c], -c))
            tokens = []
            for c in range(len(AREAS)):
                if c == 0 and not (top == 0 and p[0] >= LEVELS[0][0]):
                    continue  # "open ground" only when none dominates
                word = _level(p[c])
                if word is not None:
                    tokens.append(f"{word} {AREA_NOUNS[AREAS[c]]}")
            tokens += [f"a {name}" for k, name in enumerate(WAYS) if ways[r][k] >= WAY_MIN_CELLS]
            present = [name.replace("_", " ") for k, name in enumerate(NODES) if nodes[r][k]]
            tokens += [f"a {name}" for name in present[:MAX_NODES]]
            items += [(b, s, tok) for tok in sorted(tokens, key=_token_key)]
    return items


def render_caption(items: List[Item]) -> str:
    if not items:
        return EMPTY_CAPTION
    sentences = []
    for b in range(BANDS):
        by_sector: Dict[int, List[str]] = {}
        for bb, s, tok in items:
            if bb == b:
                by_sector.setdefault(s, []).append(tok)
        if not by_sector:
            continue
        # tokens shared by all five sectors are stated once ("all across"), the rest per sector
        common = [tok for tok in by_sector.get(0, []) if all(tok in by_sector.get(s, []) for s in range(SECTORS))]
        clauses = [", ".join(common) + " " + ALL_ACROSS] if common else []
        for s in range(SECTORS):
            rest = [tok for tok in by_sector.get(s, []) if tok not in common]
            if rest:
                clauses.append(", ".join(rest) + " " + SECTOR_PHRASES[s])
        sentences.append(f"{BAND_LABELS[b]}: " + "; ".join(clauses) + ".")
    return " ".join(sentences)


def _known_token(token: str) -> bool:
    for word in _LEVEL_RANK:
        if token.startswith(word + " "):
            return token[len(word) + 1:] in _NOUN_CLASS
    return token.startswith("a ") and (token[2:] in WAYS or token[2:].replace(" ", "_") in NODES)


def parse_caption(text: str) -> List[Item]:
    """Inverse of render_caption; raises ValueError on anything outside the grammar."""
    if text == EMPTY_CAPTION:
        return []
    if not text.endswith("."):
        raise ValueError(f"caption does not end with a period: {text!r}")
    items: List[Item] = []
    for sentence in text[:-1].split(". "):
        label, sep, body = sentence.partition(": ")
        if not sep or label not in BAND_LABELS:
            raise ValueError(f"unknown band label in {sentence!r}")
        b = BAND_LABELS.index(label)
        for clause in body.split("; "):
            sectors, toks = None, None
            if clause.endswith(" " + ALL_ACROSS):
                sectors, toks = list(range(SECTORS)), clause[: -len(ALL_ACROSS) - 1]
            else:
                for s in sorted(range(SECTORS), key=lambda s: -len(SECTOR_PHRASES[s])):
                    phrase = " " + SECTOR_PHRASES[s]
                    if clause.endswith(phrase):
                        sectors, toks = [s], clause[: -len(phrase)]
                        break
            if sectors is None:
                raise ValueError(f"no sector phrase in clause {clause!r}")
            tokens = toks.split(", ")
            for tok in tokens:
                if not _known_token(tok):
                    raise ValueError(f"unknown token {tok!r}")
            for s in sectors:
                items += [(b, s, tok) for tok in tokens]
    return sorted(items, key=lambda it: (it[0], it[1], _token_key(it[2])))



# ---------------------------------------------------------------------------
# item tensors: the content of a caption as tensors (verifier targets and mode descriptors)
# ---------------------------------------------------------------------------
LEVEL_WORDS = {1: "some", 2: "about half", 3: "mostly"}  # level index of an area class in a region (0 = not stated)
ITEM_DIM = len(AREAS) * 3 + len(WAYS) + len(NODES)  # 24 + 6 + 33 = 63


def facts_to_items(facts: dict) -> dict:
    """Tensor version of discretize for facts of any leading shape [..., R, .]:
    levels long [..., R, 8] (0 / some / about half / mostly), ways bool [..., R, 6], nodes bool [..., R, 33].
    Thresholds are compared in float64, as discretize compares Python floats."""
    p = facts["areas"].double()
    seen = (facts["visible"].double() >= VISIBLE_MIN)[..., None]
    levels = torch.zeros(p.shape, dtype=torch.long, device=p.device)
    for lvl, (thr, _) in zip((3, 2, 1), LEVELS):
        levels = torch.where((levels == 0) & (p >= thr), torch.full_like(levels, lvl), levels)
    open_ground = (p.argmax(-1) == 0) & (p[..., 0] >= LEVELS[0][0])  # "open ground" only when none dominates
    levels[..., 0] = torch.where(open_ground, torch.full_like(levels[..., 0], 3), torch.zeros_like(levels[..., 0]))
    levels = levels * seen
    ways = (facts["ways"] >= WAY_MIN_CELLS) & seen
    nodes = facts["nodes"].bool() & seen
    nodes = nodes & (nodes.long().cumsum(-1) <= MAX_NODES)
    return {"levels": levels, "ways": ways, "nodes": nodes}


def items_vector(items: dict) -> torch.Tensor:
    """[..., R, 63] float: one-hot levels 1-3 of the 8 area classes (class-major), then ways, then nodes."""
    lv = F.one_hot(items["levels"], 4)[..., 1:].flatten(-2).float()
    return torch.cat([lv, items["ways"].float(), items["nodes"].float()], -1)


def items_to_tokens(items: dict, i: int) -> set:
    """(band, sector, token) set of view i, the same tokens discretize emits (test helper)."""
    out = set()
    levels, ways, nodes = items["levels"][i].tolist(), items["ways"][i].tolist(), items["nodes"][i].tolist()
    for r in range(R):
        s, b = divmod(r, BANDS)
        out |= {(b, s, f"{LEVEL_WORDS[lv]} {AREA_NOUNS[AREAS[c]]}") for c, lv in enumerate(levels[r]) if lv}
        out |= {(b, s, f"a {WAYS[k]}") for k, on in enumerate(ways[r]) if on}
        out |= {(b, s, f"a {NODES[k].replace('_', ' ')}") for k, on in enumerate(nodes[r]) if on}
    return out

# ---------------------------------------------------------------------------
# CLI: python -m maploc.data.stage_a_captions --scene neuanspach [--split all] [--num N] [--out_dir DIR]
# ---------------------------------------------------------------------------
def build_datamodule(scene: str):
    """The scene's panorama datamodule with GT-centred +-128 m crops, no augmentation, captions loaded."""
    from omegaconf import open_dict

    from ..evaluation.eval_rho import build_cfg
    from .mapillary.dataset import MapillaryPanoDataModule

    cfg = build_cfg(scene)
    with open_dict(cfg):
        cfg.data.captions = True
        cfg.data.max_init_error = 0
        cfg.data.crop_size_meters = 2 * Z_MAX  # half-width: the 64 m disc fits for any yaw
        cfg.data.add_map_mask = False
        cfg.data.add_road_prior = False
        cfg.data.augmentation.flip = False
        cfg.data.augmentation.rot90 = False
        cfg.data.augmentation.image.apply = False
    dm = MapillaryPanoDataModule(cfg.data)
    dm.prepare_data()
    dm.setup()
    return dm


def make_loader(dm, split: str, workers: int, batch_size: int = 24):
    """Plain ordered loader (the datamodule's own train loader shuffles with a grouped sampler)."""
    from .torch import collate, worker_init_fn

    return torch.utils.data.DataLoader(
        dm.dataset(split), batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=collate, worker_init_fn=worker_init_fn, pin_memory=True,
    )


def _noun(token: str) -> str:
    for prefix in ("about half ", "some ", "mostly ", "a "):
        if token.startswith(prefix):
            return token[len(prefix):]
    return token


def main(argv=None):
    import argparse
    import json
    import random
    import sys
    import time
    from collections import Counter
    from pathlib import Path

    from tqdm import tqdm

    p = argparse.ArgumentParser(description="Stage-A reference captions from the OSM raster at the GT pose")
    p.add_argument("--scene", required=True)
    p.add_argument("--split", default="all", choices=["train", "val", "all"])
    p.add_argument("--num", type=int, default=None, help="stop after this many batches per split (partial run)")
    p.add_argument("--out_dir", type=Path, default=None, help="default: <data root>/captions")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    torch.set_num_threads(8)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = build_datamodule(args.scene)
    out_dir = args.out_dir or (dm.root / "captions")
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = args.num is not None or args.split != "all"
    stem = f"{args.scene}_stageA" + ("_partial" if partial else "")
    geo = make_geometry()

    keys = ("areas", "ways", "nodes", "visible", "camera_in_building", "camera_in_forest")
    caps, orig, names = {}, {}, []
    facts = {k: [] for k in keys + ("yaw",)}
    mismatches, t0 = 0, time.time()
    for split in ["train", "val"] if args.split == "all" else [args.split]:
        for i, data in enumerate(tqdm(make_loader(dm, split, args.workers), desc=f"stage A/{split}")):
            if args.num is not None and i >= args.num:
                break
            yaw = data["roll_pitch_yaw"][:, -1].to(dev)
            f = view_facts(data["map"].to(dev), data["uv"].to(dev), yaw, float(data["pixels_per_meter"][0]), geo)
            f = quantize_facts({k: v.cpu() for k, v in f.items()})  # render from the values the .pt will keep
            for j, name in enumerate(data["name"]):
                items = discretize(f, j)
                text = render_caption(items)
                try:
                    mismatches += parse_caption(text) != items
                except ValueError:
                    mismatches += 1
                caps[name + ".jpg"], orig[name + ".jpg"] = text, data["caption"][j]
                names.append(name + ".jpg")
            for k in keys:
                facts[k].append(f[k])
            facts["yaw"].append(yaw.cpu())
    facts = {k: torch.cat(v) for k, v in facts.items()}

    assert len(set(names)) == len(names), "duplicate view names"
    ref_path = dm.root / "captions" / f"{args.scene}.json"
    ref = json.load(open(ref_path)) if ref_path.exists() else {}
    if ref and not partial:
        assert set(caps) == set(ref), f"key sets differ: generated {len(caps)}, reference {len(ref)}"
    order = [k for k in ref if k in caps] + [k for k in caps if k not in ref]  # original file order
    perm = torch.tensor([{k: i for i, k in enumerate(names)}[k] for k in order])
    dump = {
        "names": order,
        "areas": facts["areas"][perm].half(), "ways": facts["ways"][perm].to(torch.int16),
        "nodes": facts["nodes"][perm].bool(), "visible": facts["visible"][perm].half(),
        "camera_in_building": facts["camera_in_building"][perm].bool(),
        "camera_in_forest": facts["camera_in_forest"][perm].bool(), "yaw": facts["yaw"][perm].float(),
    }
    with open(out_dir / f"{stem}.json", "w") as fp:
        json.dump({k: caps[k] for k in order}, fp, indent=2)
    torch.save(dump, out_dir / f"{stem}_facts.pt")

    # statistics
    from transformers import AutoTokenizer

    texts = [caps[k] for k in order]
    n = len(texts)
    lens = [len(ids) for ids in AutoTokenizer.from_pretrained("t5-base")(texts).input_ids]
    bands = Counter(sum(lbl in t for lbl in BAND_LABELS) for t in texts)
    mention, across = Counter(), 0
    for t in texts:
        across += ALL_ACROSS in t
        for noun in {_noun(tok) for _, _, tok in parse_caption(t)}:
            mention[noun] += 1
    vis_band = dump["visible"].float().view(n, SECTORS, BANDS).mean((0, 1))
    print(f"\n== stage A {args.scene} ({args.split}{'' if args.num is None else f', {args.num} batches'}): "
          f"{n} views in {time.time() - t0:.0f} s -> {out_dir / stem}.json / _facts.pt")
    print(f"round-trip mismatches: {mismatches}")
    print(f"bands mentioned (0/1/2/3): {[bands.get(b, 0) for b in range(4)]}  'all across' used in {across} views")
    print(f"words mean {sum(len(t.split()) for t in texts) / n:.1f}; t5 tokens mean {sum(lens) / n:.1f} max {max(lens)} "
          f"> 64 in {sum(l > 64 for l in lens) / n * 100:.1f} %")
    print("mean visible fraction per band (0-10 / 10-30 / 30-64 m): " + " / ".join(f"{v:.2f}" for v in vis_band.tolist()))
    print(f"camera in building {int(dump['camera_in_building'].sum())}, in forest {int(dump['camera_in_forest'].sum())}")
    print("mentioned in % of views: " + ", ".join(f"{k} {v / n * 100:.1f}" for k, v in mention.most_common()))
    for k in random.Random(0).sample(order, min(5, n)):
        print(f"\n[{k}]\n  VLM : {orig[k]}\n  A   : {caps[k]}")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
