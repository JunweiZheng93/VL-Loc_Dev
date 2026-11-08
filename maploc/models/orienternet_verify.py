"""OrienterNet (RHO fused_prior) + hypothesis-level text verification (design C, spec
docs/superpowers/specs/2026-11-07-language-verified-hypotheses-design.md): the image branch
proposes the top-K modes of its posterior, a caption verifier predicts the OSM area
fractions of every bearing sector x range bin and scores each mode by descriptor
similarity, and the posterior is re-weighted basin by basin (image shape inside a basin,
verifier ranking between basins)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics_v1gt import AngleError, Location2DError, MeanMetricWithRecall
from .orienternet_fused_prior import OrienterNet as OrienterNetFusedPrior
from .egocentric import NUM_AREAS, egocentric_labels
from .text_verifier import (
    RegionVerifier,
    TextEncoder,
    descriptor_similarity,
    listwise_target,
    mirror_regions,
    nms_modes,
    region_fractions,
    region_geometry,
    reweight_log_probs,
)
from .voting import argmax_xyr, expectation_xyr


class ImageLocation2DError(Location2DError):
    """Image-only read-out (what the base answers without text); distinct class so that
    torchmetrics' compute groups never merge it with the main metric."""


class ImageAngleError(AngleError):
    """Image-only read-out; distinct class as above."""


class MeanOfKey(MeanMetricWithRecall):
    """Mean of a per-sample tensor pred[key]; NaN entries (samples without a target) are skipped."""

    def __init__(self, key):
        super().__init__()
        self.key = key

    def update(self, pred, data):
        v = pred[self.key].float()
        v = v[torch.isfinite(v)]
        if v.numel():
            self.value.append(v)

    def get_errors(self):
        v = self.value
        if isinstance(v, list):
            return torch.cat(v) if v else torch.zeros(0, device=self.device)
        return v

    def compute(self):
        # no sample had a target yet (untrained model, tiny validation): NaN instead of a crash
        v = self.value
        if isinstance(v, list):
            if not v:
                return torch.tensor(float("nan"), device=self.device)
            v = torch.cat(v)
        return v.mean(0) if v.numel() else torch.tensor(float("nan"), device=self.device)


class OrienterNetVerify(OrienterNetFusedPrior):
    default_conf = {
        **OrienterNetFusedPrior.default_conf,
        "verify": {
            "K": 10,
            "K_eval": 10,
            "nms_xy": 5.0,
            "nms_yaw": 20.0,
            "sectors": 5,
            "ranges": 3,
            "coarse_cell": 2.0,
            "fov_deg": 120.0,
            "encoder": "t5-base",
            "max_tokens": 64,
            "d_model": 256,
            "layers": 2,
            "heads": 8,
            "ffn": 512,
            "tau_init": 10.0,
            "learn_beta": True,
            "alpha_d": 1.0,
            "alpha_l": 1.0,
            "blank_captions": False,
        },
    }
    TEXT_PREFIXES = ("verifier.", "log_tau", "beta", "region_logits")

    def _init(self, conf):
        super()._init(conf)
        v = conf.verify
        self.text_encoder = TextEncoder(v.encoder, v.max_tokens)
        grid, region, centres = region_geometry(v.sectors, v.ranges, conf.z_max, conf.x_max, v.coarse_cell, v.fov_deg)
        self.register_buffer("grid_c", grid, persistent=False)
        self.register_buffer("region", region, persistent=False)
        self.R = v.sectors * v.ranges
        self.verifier = RegionVerifier(
            self.text_encoder.dim, centres, d_model=v.d_model, layers=v.layers, heads=v.heads, ffn=v.ffn,
            num_classes=NUM_AREAS + 1,
        )
        self.log_tau = nn.Parameter(torch.tensor(math.log(float(v.tau_init))))
        self.beta = nn.Parameter(torch.zeros(()), requires_grad=bool(v.learn_beta))
        self.region_logits = nn.Parameter(torch.zeros(self.R))

    def text_parameter_names(self):
        """Verifier parameters (trained at training.lr_text; the base keeps training.lr)."""
        return [n for n, _ in self.named_parameters() if n.startswith(self.TEXT_PREFIXES)]

    def _descriptors(self, map_r, xy, yaw):
        """OSM area fractions per region seen from poses (xy [n, 2] px, yaw [n] deg): [n, R, C]."""
        labels = egocentric_labels(map_r, xy, yaw, self.grid_c, self.conf.pixel_per_meter)[:, 0].clamp(max=NUM_AREAS)
        onehot = F.one_hot(labels, NUM_AREAS + 1).permute(0, 3, 1, 2).float()
        return region_fractions(onehot, self.region, self.R)

    def _forward(self, data):
        pred, pred_map, scores, is_pano, extras = self._image_scores(data)
        img = self._finalize(pred, pred_map, scores, data, is_pano, extras)
        lp_img = img["log_probs"]
        step = 3 if is_pano else 1
        v = self.conf.verify
        N, ppm = self.conf.num_rotations, self.conf.pixel_per_meter
        K = v.K if self.training else v.K_eval
        r_px, r_bins = int(round(v.nms_xy * ppm)), int(round(v.nms_yaw / (360.0 / N)))
        map_r = data["map"][::step]

        with torch.no_grad():  # proposals and their map descriptors carry no gradient
            modes, logp, valid, basin = nms_modes(lp_img.detach(), K, r_px, r_bins)
            B = modes.shape[0]
            flat = modes.reshape(B * K, 3)
            d_modes = self._descriptors(
                map_r.repeat_interleave(K, 0), flat[:, :2].float(), flat[:, 2].float() * (360.0 / N)
            ).reshape(B, K, self.R, -1)
            d_gt = target = None
            if "uv" in data:
                xy_gt, yaw_gt = data["uv"][::step], data["roll_pitch_yaw"][::step, -1]
                d_gt = self._descriptors(map_r, xy_gt, yaw_gt)
                target = listwise_target(modes, valid, xy_gt, yaw_gt, N, ppm, v.nms_xy, v.nms_yaw)

        captions = list(data["caption"][::step])
        if v.blank_captions:
            captions = [""] * len(captions)
        tokens, mask = self.text_encoder(captions)
        log_d_hat = F.log_softmax(self.verifier(tokens, mask), -1)
        if "flip" in data:  # mirrored training image: map and pose live in the mirrored frame, the caption does not
            flip = data["flip"][::step].bool()
            log_d_hat = torch.where(flip[:, None, None], mirror_regions(log_d_hat, v.sectors, v.ranges), log_d_hat)
        d_hat = log_d_hat.exp()

        sim = descriptor_similarity(d_hat, d_modes, self.region_logits)
        dl = torch.where(valid, logp - logp[:, :1], torch.zeros_like(logp))
        s = (torch.exp(self.log_tau) * sim + self.beta * dl).masked_fill(~valid, float("-inf"))
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
            "d_hat": d_hat,
            "log_d_hat": log_d_hat,
            "d_modes": d_modes,
        }
        if d_gt is not None:
            has = target >= 0
            sel = log_q.argmax(1)
            nan = torch.full_like(has, float("nan"), dtype=torch.float)
            out.update(
                d_gt=d_gt,
                target=target,
                recall_hit=has.float(),
                select_hit=torch.where(has, (sel == target).float(), nan),
            )
        return out

    def loss(self, pred, data):
        # image NLL on the pure image posterior: the image branch trains exactly as the RHO baseline
        losses = super().loss({**pred, "log_probs": pred["log_probs_img"]}, data)
        v = self.conf.verify
        desc_ce = -(pred["d_gt"] * pred["log_d_hat"]).sum(-1).mean(-1)  # soft-target CE, mean over regions
        target = pred["target"]
        has = target >= 0
        picked = -pred["log_q"].gather(1, target.clamp(min=0)[:, None])[:, 0]
        listwise = torch.where(has, picked, torch.zeros_like(picked))
        losses["desc_ce"], losses["listwise"] = desc_ce, listwise
        if v.alpha_d:
            losses["total"] = losses["total"] + v.alpha_d * desc_ce
        if v.alpha_l:
            losses["total"] = losses["total"] + v.alpha_l * listwise
        return losses

    def metrics(self):
        return {
            **super().metrics(),
            "xy_max_error_img": ImageLocation2DError("uv_max_img", self.conf.pixel_per_meter),
            "yaw_max_error_img": ImageAngleError("yaw_max_img"),
            "select_acc": MeanOfKey("select_hit"),
            "mode_recall": MeanOfKey("recall_hit"),
        }
