"""OrienterNet (RHO fused_prior) + three-caption item verification (spec
docs/superpowers/specs/2026-11-08-item-verifier-design.md): design C with a verifier whose target
matches the stage-A captions. For each of the three view captions a region-query head predicts the
stage-A items (5 sectors x 3 bands, 0-64 m: coverage level of every area class, roads, point features);
every top-K mode of the image posterior is scored by the soft Jaccard between the predicted items and
the items the stage-A code (maploc/data/stage_a_captions.py) computes at that mode on the uncut
map_desc raster. Proposals, listwise loss and basin re-weighting are those of design C."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import stage_a_captions as SA
from .orienternet_fused_prior import OrienterNet as OrienterNetFusedPrior
from .orienternet_verify import OrienterNetVerify
from .text_verifier import RegionVerifier, TextEncoder, listwise_target, mirror_regions, nms_modes, reweight_log_probs
from .voting import argmax_xyr, expectation_xyr

OFF = (0.0, -120.0, 120.0)  # yaw of view1 / view2 / view3 relative to view1 (deg, clockwise), measured on val
N_LEVELS = 4  # not stated / some / about half / mostly
N_AREA_LOGITS = len(SA.AREAS) * N_LEVELS  # 32
HEAD_DIM = N_AREA_LOGITS + len(SA.WAYS) + len(SA.NODES)  # 71


def item_soft_jaccard(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """p [B,V,R,D] item probabilities, y [B,K,V,R,D] 0/1 items of every mode -> [B,K]
    sum p*y / sum (y + p*(1-y)): the set Jaccard when p is 0/1."""
    p = p[:, None]
    inter = (p * y).flatten(2).sum(-1)
    union = (y + p * (1 - y)).flatten(2).sum(-1)
    return inter / union.clamp(min=1e-6)


def mirror_sectors(x: torch.Tensor) -> torch.Tensor:
    """Left-right mirror of stage-A regions [..., R, C] (region = sector * 3 + band): sector s <-> 4 - s."""
    return mirror_regions(x, SA.SECTORS, SA.BANDS)


def item_probs(logits: torch.Tensor) -> torch.Tensor:
    """Head logits [..., 71] -> item probabilities [..., 63] in the order of stage_a_captions.items_vector."""
    lv = logits[..., :N_AREA_LOGITS].unflatten(-1, (len(SA.AREAS), N_LEVELS)).softmax(-1)[..., 1:].flatten(-2)
    ways = logits[..., N_AREA_LOGITS:N_AREA_LOGITS + len(SA.WAYS)].sigmoid()
    nodes = logits[..., N_AREA_LOGITS + len(SA.WAYS):].sigmoid()
    return torch.cat([lv, ways, nodes], -1)


class OrienterNetVerifyItems(OrienterNetVerify):
    default_conf = {
        **OrienterNetVerify.default_conf,
        "verify": {**OrienterNetVerify.default_conf["verify"], "max_tokens": 192, "desc_offsets": list(OFF)},
    }
    TEXT_PREFIXES = ("verifier.", "log_tau", "beta")

    def _init(self, conf):
        OrienterNetFusedPrior._init(self, conf)  # image branch only; the verifier differs from design C
        v = conf.verify
        self.text_encoder = TextEncoder(v.encoder, v.max_tokens)
        self.geo = SA.make_geometry()  # CPU tensors, moved per call by stage_a_captions
        x, z = self.geo.grid[..., 0], self.geo.grid[..., 1]
        phi, rr = torch.atan2(x, z), torch.sqrt(x**2 + z**2) / SA.Z_MAX
        centres = torch.stack([torch.stack([phi[self.geo.region == r].mean(), rr[self.geo.region == r].mean()]) for r in range(SA.R)])
        self.verifier = RegionVerifier(
            self.text_encoder.dim, centres, d_model=v.d_model, layers=v.layers, heads=v.heads, ffn=v.ffn,
            num_classes=HEAD_DIM,
        )
        self.log_tau = nn.Parameter(torch.tensor(math.log(float(v.tau_init))))
        self.beta = nn.Parameter(torch.zeros(()), requires_grad=bool(v.learn_beta))

    def _captions(self, data):
        caps = list(data["caption"])  # 3 per panorama, view1 / view2 / view3
        return [""] * len(caps) if self.conf.verify.blank_captions else caps

    def _flip(self, data, B, device):
        if "flip" in data:
            return data["flip"][::3].bool()
        return torch.zeros(B, dtype=torch.bool, device=device)

    def _items_at(self, data, uv1, yaw1):
        """Stage-A items of the three views seen from poses of view1: uv1 [B,M,2] model-crop px, yaw1 [B,M] deg.
        Returns levels / ways / nodes shaped [B, M, 3, R, .] (no gradient)."""
        map_d = data["map_desc"][::3].long()
        B, M = uv1.shape[:2]
        o = (map_d.shape[-1] - data["map"].shape[-1]) / 2
        off = torch.tensor(list(self.conf.verify.desc_offsets), device=uv1.device, dtype=yaw1.dtype)
        sign = torch.where(self._flip(data, B, uv1.device), -1.0, 1.0).to(yaw1)[:, None, None]
        yaws = (yaw1[..., None] + sign * off).reshape(B, M * 3)  # mirrored frame: relative yaws change sign
        uv = (uv1 + o)[:, :, None].expand(-1, -1, 3, -1).reshape(B, M * 3, 2)
        f = SA.quantize_facts(SA.view_facts_multi(map_d, uv, yaws, self.conf.pixel_per_meter, self.geo))
        items = SA.facts_to_items(f)
        return {k: t.reshape(B, M, 3, *t.shape[2:]) for k, t in items.items()}

    def _forward(self, data):
        pred, pred_map, scores, is_pano, extras = self._image_scores(data)
        img = self._finalize(pred, pred_map, scores, data, is_pano, extras)
        assert is_pano, "the item verifier needs panorama batches (3 views per sample)"
        lp_img = img["log_probs"]
        v = self.conf.verify
        N, ppm = self.conf.num_rotations, self.conf.pixel_per_meter
        K = v.K if self.training else v.K_eval
        r_px, r_bins = int(round(v.nms_xy * ppm)), int(round(v.nms_yaw / (360.0 / N)))

        with torch.no_grad():  # proposals and their items carry no gradient
            modes, logp, valid, basin = nms_modes(lp_img.detach(), K, r_px, r_bins)
            B = modes.shape[0]
            uv_m, yaw_m = modes[..., :2].float(), modes[..., 2].float() * (360.0 / N)
            has_gt = "uv" in data
            if has_gt:
                xy_gt, yaw_gt = data["uv"][::3], data["roll_pitch_yaw"][::3, -1]
                uv_m, yaw_m = torch.cat([xy_gt[:, None], uv_m], 1), torch.cat([yaw_gt[:, None], yaw_m], 1)
            items = self._items_at(data, uv_m, yaw_m)
            y_all = SA.items_vector(items)  # [B, (1 +) K, 3, R, 63]
            y_modes = y_all[:, 1:] if has_gt else y_all

        tokens, mask = self.text_encoder(self._captions(data))
        logits = self.verifier(tokens, mask).reshape(B, 3, SA.R, HEAD_DIM)
        flip = self._flip(data, B, logits.device)
        # mirrored training sample: map and poses live in the mirrored frame, the captions do not
        logits = torch.where(flip[:, None, None, None], mirror_sectors(logits), logits)
        p_items = item_probs(logits)
        J = item_soft_jaccard(p_items, y_modes)
        dl = torch.where(valid, logp - logp[:, :1], torch.zeros_like(logp))
        s = (torch.exp(self.log_tau) * J + self.beta * dl).masked_fill(~valid, float("-inf"))
        log_q = F.log_softmax(s, 1)
        lp = reweight_log_probs(lp_img.detach(), basin, logp, log_q.detach(), valid)
        with torch.no_grad():
            uvr = argmax_xyr(lp).to(lp)
            uvr_avg, _ = expectation_xyr(lp.exp())
        out = {
            **img,
            "log_probs": lp,
            "log_probs_img": lp_img,
            "uvr_max": uvr,
            "uv_max": uvr[..., :2],
            "yaw_max": uvr[..., 2],
            "uvr_expectation": uvr_avg,
            "uv_expectation": uvr_avg[..., :2],
            "yaw_expectation": uvr_avg[..., 2],
            "uv_max_img": img["uv_max"],
            "yaw_max_img": img["yaw_max"],
            "modes": modes,
            "modes_valid": valid,
            "modes_logp": logp,
            "log_q": log_q,
            "item_logits": logits,
            "p_items": p_items,
            "y_modes": y_modes,
            "jaccard": J,
        }
        if has_gt:
            target = listwise_target(modes, valid, xy_gt, yaw_gt, N, ppm, v.nms_xy, v.nms_yaw)
            has = target >= 0
            sel = log_q.argmax(1)
            nan = torch.full_like(has, float("nan"), dtype=torch.float)
            out.update(
                y_gt={k: t[:, 0] for k, t in items.items()},
                target=target,
                recall_hit=has.float(),
                select_hit=torch.where(has, (sel == target).float(), nan),
            )
        return out

    def loss(self, pred, data):
        # image NLL on the pure image posterior: the image branch trains exactly as the RHO baseline
        losses = OrienterNetFusedPrior.loss(self, {**pred, "log_probs": pred["log_probs_img"]}, data)
        v = self.conf.verify
        lg, y = pred["item_logits"], pred["y_gt"]
        lv = lg[..., :N_AREA_LOGITS].unflatten(-1, (len(SA.AREAS), N_LEVELS))
        ce = F.cross_entropy(lv.flatten(0, -2), y["levels"].flatten(), reduction="none").view(y["levels"].shape).mean(-1)
        bw = F.binary_cross_entropy_with_logits(
            lg[..., N_AREA_LOGITS:N_AREA_LOGITS + len(SA.WAYS)], y["ways"].float(), reduction="none").mean(-1)
        bn = F.binary_cross_entropy_with_logits(
            lg[..., N_AREA_LOGITS + len(SA.WAYS):], y["nodes"].float(), reduction="none").mean(-1)
        item = (ce + bw + bn).mean((1, 2))  # [B] mean over views x regions
        target = pred["target"]
        has = target >= 0
        picked = -pred["log_q"].gather(1, target.clamp(min=0)[:, None])[:, 0]
        listwise = torch.where(has, picked, torch.zeros_like(picked))
        losses["desc_ce"], losses["listwise"] = item, listwise  # "desc_ce" keeps the design-C log tag
        if v.alpha_d:
            losses["total"] = losses["total"] + v.alpha_d * item
        if v.alpha_l:
            losses["total"] = losses["total"] + v.alpha_l * listwise
        return losses
