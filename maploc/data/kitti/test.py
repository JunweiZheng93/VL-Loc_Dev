import torch
import os
import glob
from diffusers.utils import load_image
from diffusers import StableDiffusionXLPipeline, EDMEulerScheduler, StableDiffusionXLInstructPix2PixPipeline, AutoencoderKL
import numpy as np
import math
from PIL import Image

edit_file_local_path = "/hkfs/work/workspace/scratch/tj3409-rdaionet/models/cosxl/cosxl_edit.safetensors"
vae_local_path = "/hkfs/work/workspace/scratch/tj3409-rdaionet/models/sdxl-vae-fp16-fix"

# Make sure the width and height are multiples of 8
def resize_image(image):
    
    original_width, original_height = image.size
    new_width = original_width - original_width % 8
    new_height = original_height - original_height % 8
    
    resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_img


vae = AutoencoderKL.from_pretrained(vae_local_path, torch_dtype=torch.float16)
generator = torch.Generator("cuda")
pipe_edit = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
    edit_file_local_path, num_in_channels=8, is_cosxl_edit=True, vae=vae, torch_dtype=torch.float16,
)
pipe_edit.scheduler = EDMEulerScheduler(sigma_min=0.002, sigma_max=120.0, sigma_data=1.0, prediction_type="v_prediction", sigma_schedule="exponential")
pipe_edit.to("cuda")
pipe_edit.enable_model_cpu_offload()

# --- input directory ---
input_path = "/hkfs/work/workspace/scratch/tj3409-rdaionet/OrienterNet/datasets/kitti/2011_10_03/2011_10_03_drive_0027_sync/image_02/data/0000000000.png"

# --- output directory ---
output_base_dir = "/hkfs/work/workspace/scratch/tj3409-rdaionet/OrienterNet/datasets/weather_noise"
os.makedirs(output_base_dir, exist_ok=True) # check the output directory

# --- Parameters ---
guidance_scale=6.75
image_guidance_scale=2.5
num_inference_steps=100

prompt = "Transform it into a heavy foggy day. Create a diffused lighting effect. Add mist on the scene."
negative_prompt = "low quality, bad quality, worst quality, sketches, noisy, blurry, jpeg artifacts, deformed, fused objects, poorly drawn features"




print(f"--- Processing Input Image: {input_path} ---")
image = load_image(input_path)
original_width, original_height = image.size
image = resize_image(image)
new_width, new_height = image.size

current_seed = 18457
generator.manual_seed(current_seed)

edited_image = pipe_edit(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image,
    height=new_height,
    width=new_width,
    guidance_scale=guidance_scale,
    image_guidance_scale=image_guidance_scale,
    num_inference_steps=num_inference_steps,
    generator=generator,
).images

# --- Save ouput ---
output_filename = "test.png"
output_path = os.path.join(output_base_dir, output_filename)
edited_image[0].save(output_path)
print(f"  Saved: {output_path}")

print("--- Processing Complete ---")

