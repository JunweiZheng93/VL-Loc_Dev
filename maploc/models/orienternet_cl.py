# Copyright (c) Meta Platforms, Inc. and affiliates.
import warnings
import numpy as np
import torch
from torch.nn.functional import normalize
from omegaconf import OmegaConf

from . import get_model
from .infonce_loss import multi_positive_infonce_loss, multi_positive_infonce_loss_vectorized, safe_multi_positive_infonce_loss_vectorized
from .base import BaseModel
from .bev_net import BEVNet
from .bev_projection import CartesianProjection, PolarProjectionDepth
from .map_encoder import MapEncoder
from .metrics import AngleError, AngleRecall, Location2DError, Location2DRecall
from .voting import (
    TemplateSampler,
    argmax_xyr,
    conv2d_fft_batchwise,
    expectation_xyr,
    log_softmax_spatial,
    mask_yaw_prior,
    nll_loss_xyr,
    nll_loss_xyr_smoothed,
)


class OrienterNet_CL(BaseModel):
    default_conf = {
        "image_encoder": "???",
        "map_encoder": "???",
        "bev_net": "???",
        "latent_dim": "???",
        "matching_dim": "???",
        "scale_range": [0, 9],
        "num_scale_bins": "???",
        "z_min": None,
        "z_max": "???",
        "x_max": "???",
        "pixel_per_meter": "???",
        "num_rotations": "???",
        "add_temperature": False,
        "normalize_features": False,
        "padding_matching": "replicate",
        "apply_map_prior": True,
        "do_label_smoothing": False,
        "sigma_xy": 1,
        "sigma_r": 2,
        # depcreated
        "depth_parameterization": "scale",
        "norm_depth_scores": False,
        "normalize_scores_by_dim": False,
        "normalize_scores_by_num_valid": True,
        "prior_renorm": True,
        "retrieval_dim": None,
        "contrastive_loss_weight": 0.0,
        "contrastive_temperature": 0.1,
        "embedding_monitor_interval": 50,
    }
    required_data_keys = []

    def _init(self, conf):
        assert not self.conf.norm_depth_scores
        assert self.conf.depth_parameterization == "scale"
        assert not self.conf.normalize_scores_by_dim
        assert self.conf.normalize_scores_by_num_valid
        assert self.conf.prior_renorm

        Encoder = get_model(conf.image_encoder.get("name", "feature_extractor_v2"))
        # 1. Create a new, plain Python dictionary from the backbone config.
        #    This is mutable and detached from the read-only parent.
        encoder_conf_dict = OmegaConf.to_container(conf.image_encoder.backbone, resolve=True)
        
        # 2. Safely get the global_descriptor config from the parent level.
        gd_conf = conf.image_encoder.get('global_descriptor')
        
        # 3. If it exists, add it to our new dictionary.
        if gd_conf:
            encoder_conf_dict['global_descriptor'] = OmegaConf.to_container(gd_conf, resolve=True)
        
        # 4. Create a fresh OmegaConf object from our new dictionary.
        final_encoder_conf = OmegaConf.create(encoder_conf_dict)
        
        # 5. Initialize the encoder with this new, correct config object.
        self.image_encoder = Encoder(final_encoder_conf)
        self.map_encoder = MapEncoder(conf.map_encoder)
        self.bev_net = None if conf.bev_net is None else BEVNet(conf.bev_net)

        ppm = conf.pixel_per_meter
        self.projection_polar = PolarProjectionDepth(
            conf.z_max,
            ppm,
            conf.scale_range,
            conf.z_min,
        )
        self.projection_bev = CartesianProjection(
            conf.z_max, conf.x_max, ppm, conf.z_min
        )
        self.template_sampler = TemplateSampler(
            self.projection_bev.grid_xz, ppm, conf.num_rotations
        )

        self.scale_classifier = torch.nn.Linear(conf.latent_dim, conf.num_scale_bins)
        if conf.bev_net is None:
            self.feature_projection = torch.nn.Linear(
                conf.latent_dim, conf.matching_dim
            )
        if conf.add_temperature:
            temperature = torch.nn.Parameter(torch.tensor(0.0))
            self.register_parameter("temperature", temperature)

    def exhaustive_voting(self, f_bev, f_map, valid_bev, confidence_bev=None):
        if self.conf.normalize_features:
            f_bev = normalize(f_bev, dim=1)
            f_map = normalize(f_map, dim=1)

        # Build the templates and exhaustively match against the map.
        if confidence_bev is not None:
            f_bev = f_bev * confidence_bev.unsqueeze(1)
        f_bev = f_bev.masked_fill(~valid_bev.unsqueeze(1), 0.0)
        templates = self.template_sampler(f_bev)
        with torch.autocast("cuda", enabled=False):
            scores = conv2d_fft_batchwise(
                f_map.float(),
                templates.float(),
                padding_mode=self.conf.padding_matching,
            )
        if self.conf.add_temperature:
            scores = scores * torch.exp(self.temperature)

        # Reweight the different rotations based on the number of valid pixels in each
        # template. Axis-aligned rotation have the maximum number of valid pixels.
        valid_templates = self.template_sampler(valid_bev.float()[None]) > (1 - 1e-4)
        num_valid = valid_templates.float().sum((-3, -2, -1))
        scores = scores / num_valid[..., None, None]
        return scores

    def _forward(self, data: dict) -> dict:
        """
        Forward pass of the model.
        - For standard localization, it processes only the anchor images.
        - If in training mode, it also computes the contrastive loss between
          anchors and their corresponding positive (noisy) samples.
        """
        pred = {}
        pred_map = pred["map"] = self.map_encoder(data)
        f_map = pred_map["map_features"][0]

        # The first B samples in the batch are the anchors.
        num_anchors = data['camera'].f.shape[0]
        # num_anchors = data['num_anchors']


        # 1. ENCODE ALL IMAGES (ANCHORS + POSITIVES)
        # data['image'] is a stacked tensor of [Anchors, Positives_1, Positives_2, ...]
        # Its total size is B * (1 + P), where B=batch_size, P=num_positives.
        all_image_feats = self.image_encoder(data)

        # 2. STANDARD ORIENTERNET LOCALIZATION PIPELINE (ANCHORS ONLY)
        # Select features for anchor images
        level = 0
        anchor_f_image = all_image_feats['feature_maps'][level][:num_anchors]
        camera = data["camera"].scale(1 / self.image_encoder.scales[level])
        camera = camera.to(data["image"].device, non_blocking=True)

        # Estimate the monocular priors.
        pred["pixel_scales"] = scales = self.scale_classifier(anchor_f_image.moveaxis(1, -1))
        f_polar = self.projection_polar(anchor_f_image, scales, camera)

        # Map to the BEV.
        with torch.autocast("cuda", enabled=False):
            f_bev, valid_bev, _ = self.projection_bev(
                f_polar.float(), None, camera.float()
            )
        pred_bev = {}
        if self.conf.bev_net is None:
            # channel last -> classifier -> channel first
            f_bev = self.feature_projection(f_bev.moveaxis(1, -1)).moveaxis(-1, 1)
        else:
            pred_bev = pred["bev"] = self.bev_net({"input": f_bev})
            f_bev = pred_bev["output"]
        
        # Map features and exhaustive voting
        scores = self.exhaustive_voting(
            f_bev, f_map, valid_bev, pred_bev.get("confidence")
        )
        scores = scores.moveaxis(1, -1) # B,H,W,N

        if "log_prior" in pred_map and self.conf.apply_map_prior:
            scores = scores + pred_map["log_prior"][0].unsqueeze(-1)
        if "map_mask" in data:
            scores.masked_fill_(~data["map_mask"][..., None], -np.inf)
        if "yaw_prior" in data:
            mask_yaw_prior(scores, data["yaw_prior"], self.conf.num_rotations)

        log_probs = log_softmax_spatial(scores)
        with torch.no_grad():
            uvr_max = argmax_xyr(scores).to(scores)
            uvr_avg, _ = expectation_xyr(log_probs.exp())

        pred.update({
            "scores": scores, "log_probs": log_probs,
            "uvr_max": uvr_max, "uv_max": uvr_max[..., :2], "yaw_max": uvr_max[..., 2],
            "uvr_expectation": uvr_avg, "uv_expectation": uvr_avg[..., :2], "yaw_expectation": uvr_avg[..., 2],
            "features_image": anchor_f_image, "features_bev": f_bev, "valid_bev": valid_bev.squeeze(1),
        })

        # 3. Loss Calculation
        # First, get the standard localization loss
        pred['losses'] = self.loss(pred, data)
        # Then, if in training, compute and add the contrastive loss
        global_descriptors = all_image_feats.get("global_descriptor")
        if self.training and global_descriptors is not None:
            positives_per_anchor = data['positives_per_anchor']
            num_positives = data['num_positives']            
            anchor_embeddings = global_descriptors[:num_anchors]
            positive_embeddings_flat = global_descriptors[num_anchors: num_anchors+ num_positives]
            embed_dim = positive_embeddings_flat.shape[1]
            positive_embeddings = positive_embeddings_flat.view(
                num_anchors, positives_per_anchor, embed_dim)
            
            contrastive_loss = multi_positive_infonce_loss_vectorized(
                anchor_embeddings,
                positive_embeddings,
                temperature=self.conf.contrastive_temperature)
            
            weighted_contrastive = contrastive_loss * self.conf.contrastive_loss_weight
            pred['losses']['contrastive_loss'] = weighted_contrastive
            # Add to the 'total' loss for optimization
            pred['losses']['total'] = pred['losses']['contrastive_loss'] + pred['losses']['nll']
        else:
            warnings.warn(
                "contrastive_loss_weight > 0 but 'global_descriptor' not found."
            )

        return pred

    def loss(self, pred, data):
        xy_gt = data["uv"]
        yaw_gt = data["roll_pitch_yaw"][..., -1]
        if self.conf.do_label_smoothing:
            nll = nll_loss_xyr_smoothed(
                pred["log_probs"],
                xy_gt,
                yaw_gt,
                self.conf.sigma_xy / self.conf.pixel_per_meter,
                self.conf.sigma_r,
                mask=data.get("map_mask"),
            )
        else:
            nll = nll_loss_xyr(pred["log_probs"], xy_gt, yaw_gt)
        loss = {"total": nll, "nll": nll}
        if self.training and self.conf.add_temperature:
            loss["temperature"] = self.temperature.expand(len(nll))
        return loss

    def metrics(self):
        return {
            "xy_max_error": Location2DError("uv_max", self.conf.pixel_per_meter),
            "xy_expectation_error": Location2DError(
                "uv_expectation", self.conf.pixel_per_meter
            ),
            "yaw_max_error": AngleError("yaw_max"),
            "xy_recall_2m": Location2DRecall(2.0, self.conf.pixel_per_meter, "uv_max"),
            "xy_recall_5m": Location2DRecall(5.0, self.conf.pixel_per_meter, "uv_max"),
            "yaw_recall_2°": AngleRecall(2.0, "yaw_max"),
            "yaw_recall_5°": AngleRecall(5.0, "yaw_max"),
        }


