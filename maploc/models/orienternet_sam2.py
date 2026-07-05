import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .orienternet import OrienterNet
from .voting import mask_yaw_prior, log_softmax_spatial, argmax_xyr, expectation_xyr

from sam2.build_sam import build_sam2

class OrienterNetWithSAM2Encoder(OrienterNet):
    """
    OrienterNet model using a frozen SAM2 image encoder.
    """
    def _init(self, conf):
        # Use the -init() from OrienterNet to construct a complete model
        super()._init(conf)
        
        print("Loading SAM2 model...")
        # Check the correct path to SAM2 config according to your config search path
        sam2_cfg_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
        # sam2_cfg_path = "sam2.1_hiera_l.yaml"
        sam2_ckpt_path = "/hkfs/work/workspace/scratch/tj3409-rdaionet/sam2/checkpoints/sam2.1_hiera_large.pt"
        
        sam2_model = build_sam2(sam2_cfg_path, sam2_ckpt_path)
        
        # Replace the image encoder with SAM2's encoder
        self.image_encoder = sam2_model.image_encoder
        print("Successfully replaced OrienterNet image encoder with SAM2's.")

        # Freeze the weights of the SAM2 encoder to avoid catastrophic forgetting
        print("Freezing weights of the SAM2 image encoder at first...")
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze weights at top layers seletively
        num_blocks = 48
        num_blocks_to_unfreeze = 12
        unfreeze_start_index = num_blocks - num_blocks_to_unfreeze

        print(f"Unfreeze the last {num_blocks_to_unfreeze} Trunk blocks and entire Neck...")

        for name, param in self.image_encoder.named_parameters():
            if 'neck.' in name:
                param.requires_grad=True
                print(f"Unfreeze param: {name}")
                continue
            
            if 'trunk.blocks.'in name:
                try:
                    block_num_str = name.split('.')[2]
                    if block_num_str.isdigit():
                        block_num = int(block_num_str)
                        if block_num >= unfreeze_start_index:
                            param.requires_grad = True
                            print(f"Unfreeze Trunk Block {block_num} : {name}")
                except (IndexError, ValueError):
                    continue

        self.image_encoder.eval()

        # Manually define the output scales of the SAM2 FPN
        # Verify using verify_scales.py
        self.image_encoder.scales = [4, 8, 16]

        # Create an adaptation layer for feature dimension mismatch
        # SAM2's encoder (Hiera-L) outputs 256-dim features.
        # OrienterNet's BEV network expects `latent_dim` (128) features.
        sam2_feature_dim = 256  # For Hiera models
        orienternet_latent_dim = conf.latent_dim
        '''
        self.encoder_adaptation = nn.Conv2d(
            sam2_feature_dim, orienternet_latent_dim, kernel_size=1
        )
        '''
        self.encoder_adaptation = nn.Sequential(
            nn.Conv2d(sam2_feature_dim, orienternet_latent_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(orienternet_latent_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(orienternet_latent_dim, orienternet_latent_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(orienternet_latent_dim),
        )

        self.adapter_shortcut = nn.Conv2d(sam2_feature_dim, orienternet_latent_dim, kernel_size=1)

    def _forward(self, data):
        # This method overrides the original _forward to handle the new encoder
        
        pred = {}
        pred_map = pred["map"] = self.map_encoder(data)
        f_map = pred_map["map_features"][0]

        # --- MODIFIED PART START ---
        
        # 1. Use the SAM2 encoder and get the finest feature map (level 0)
        # The SAM2 encoder returns features from finest to coarsest resolution.
        # The output key for FPN features in SAM2 is "backbone_fpn".
        level = 0
        # The SAM2 encoder expects a normalized image, but this is handled internally.
        f_image_raw = self.image_encoder(data['image'])["backbone_fpn"][level]
        
        # 2. Adapt the feature dimension with Residual Connection
        identity = self.adapter_shortcut(f_image_raw)
        f_image_adapted = self.encoder_adaptation(f_image_raw)
        f_image = torch.relu(identity + f_image_adapted)

        # 3. Scale the camera using the manually defined scales
        camera = data["camera"].scale(1 / self.image_encoder.scales[level])
        camera = camera.to(data["image"].device, non_blocking=True)
        
        # --- MODIFIED PART END ---

        # Estimate the monocular priors.
        pred["pixel_scales"] = scales = self.scale_classifier(f_image.moveaxis(1, -1))
        f_polar = self.projection_polar(f_image, scales, camera)

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

        scores = self.exhaustive_voting(
            f_bev, f_map, valid_bev, pred_bev.get("confidence")
        )
        scores = scores.moveaxis(1, -1)  # B,H,W,N
        if "log_prior" in pred_map and self.conf.apply_map_prior:
            scores = scores + pred_map["log_prior"][0].unsqueeze(-1)
        # pred["scores_unmasked"] = scores.clone()
        if "map_mask" in data:
            scores.masked_fill_(~data["map_mask"][..., None], -np.inf)
        if "yaw_prior" in data:
            mask_yaw_prior(scores, data["yaw_prior"], self.conf.num_rotations)
        log_probs = log_softmax_spatial(scores)
        with torch.no_grad():
            uvr_max = argmax_xyr(scores).to(scores)
            uvr_avg, _ = expectation_xyr(log_probs.exp())

        return {
            **pred,
            "scores": scores,
            "log_probs": log_probs,
            "uvr_max": uvr_max,
            "uv_max": uvr_max[..., :2],
            "yaw_max": uvr_max[..., 2],
            "uvr_expectation": uvr_avg,
            "uv_expectation": uvr_avg[..., :2],
            "yaw_expectation": uvr_avg[..., 2],
            "features_image": f_image,
            "features_bev": f_bev,
            "valid_bev": valid_bev.squeeze(1),
        }
