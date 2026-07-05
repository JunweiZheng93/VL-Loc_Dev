import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import os

from maploc.models.orienternet_sam2 import OrienterNetWithSAM2Encoder
from maploc.models.orienternet_cl import OrienterNet_CL
from maploc.module import GenericModule

from hydra import compose, initialize
from hydra.utils import instantiate

def main():
    onet_sam2_ckpt_path = "/hkfs/work/workspace/scratch/tj3409-rdaionet/OrienterNet/experiments/OrienterNet_SAM2_KITTI_mix/checkpoint-epoch=04.ckpt"
    onet_sam2 = torch.load(onet_sam2_ckpt_path, map_location="cpu", weights_only=False)

    onet_cl_ckpt_path = "/hkfs/work/workspace/scratch/tj3409-rdaionet/OrienterNet/experiments/OrienterNet_KITTI_CL1/checkpoint-step=20000.ckpt"
    onet_cl = torch.load(onet_cl_ckpt_path, map_location="cpu", weights_only=False)

    onet_ckpt_path = "/hkfs/work/workspace/scratch/tj3409-rdaionet/OrienterNet/experiments/orienternet_mgl.ckpt"
    onet = torch.load(onet_ckpt_path, map_location="cpu", weights_only=False)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "onet_strc.txt")
  

    with open(output_file, 'w', encoding='utf-8') as f:
      f.write(f"Available keys in onet checkpoint: {list(onet.keys())}\n\n")
      if "state_dict" in list(onet.keys()):
        f.write("Available keys in onet state dict:\n\n")
        for key, value in onet["state_dict"].items():
          f.write(f"Key:{key}\n")
      else:
        f.write("No state dict found in onet.\n")
      
      f.write("="*80+"\n\n")

      f.write(f"Available keys in onet_sam2 checkpoint: {list(onet_sam2.keys())}\n\n")
      if "state_dict" in list(onet_sam2.keys()):
        f.write("Available keys in onet_sam2 state dict:\n\n")
        for key, value in onet_sam2["state_dict"].items():
          f.write(f"Key:{key}\n")
      else:
        f.write("No state dict found in onet_sam2")
      
      f.write("="*80+"\n\n")

      f.write(f"Available keys in onet_cl checkpoint: {list(onet_cl.keys())}\n\n")
      if "state_dict" in list(onet_cl.keys()):
        f.write("Available keys in onet_cl state dict:\n\n")
        for key, value in onet_cl["state_dict"].items():
          f.write(f"Key:{key}\n")
      else:
        f.write("No state dict found in onet_cl")


if __name__ == '__main__':
    main()