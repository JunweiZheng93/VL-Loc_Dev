#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build KITTI dataset weather perturbations variants (Distributed Processing Version)
"""
import torch
import os
import glob
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from diffusers.utils import load_image
from diffusers import StableDiffusionXLPipeline, EDMEulerScheduler, StableDiffusionXLInstructPix2PixPipeline, AutoencoderKL
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError

# --- Global Model Setup ---
# This will be loaded by each independent process/task

edit_file_local_path = "/hkfs/work/workspace/scratch/tj3409-rdaionet/models/cosxl/cosxl_edit.safetensors"
vae_local_path = "/hkfs/work/workspace/scratch/tj3409-rdaionet/models/sdxl-vae-fp16-fix"

try:
    print(f"Process {os.getpid()}: Loading diffusion model...")
    vae = AutoencoderKL.from_pretrained(vae_local_path, torch_dtype=torch.float16)
    pipe_edit = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
        edit_file_local_path, num_in_channels=8, is_cosxl_edit=True, vae=vae, torch_dtype=torch.float16,
    )
    pipe_edit.scheduler = EDMEulerScheduler(sigma_min=0.002, sigma_max=120.0, sigma_data=1.0, prediction_type="v_prediction", sigma_schedule="exponential")
    pipe_edit.to("cuda") # Each process gets its own CUDA context from Slurm/srun
    generator = torch.Generator("cuda")
    print(f"Process {os.getpid()}: Model loaded successfully to CUDA device: {torch.cuda.current_device()}.")
except Exception as e:
    print(f"Process {os.getpid()}: Error loading model: {e}")
    exit(1)

# --- Resize Function (Uses PIL) ---
def resize_image(image: Image.Image) -> Image.Image:
    """Resizes a PIL image so its width and height are multiples of 8."""
    original_width, original_height = image.size
    new_width = original_width - original_width % 8
    new_height = original_height - original_height % 8
    if new_width <= 0 or new_height <= 0:
        print(f"Warning: Original image size ({original_width}x{original_height}) results in non-positive dimensions after rounding down to multiple of 8 ({new_width}x{new_height}). Skipping resize.")
        return image
    resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_img

# --- Prompts ---
rainy_prompt = "Transform it into a rainy day. Add visible falling rain streaks. Make surfaces appear wet and reflective. Create a soft, diffused overcast lighting effect."
night_prompt = "Transform it into deep dark night. Make the sky and street dark. Create stark, high-contrast lighting effect."
foggy_prompt = "Transform it into a heavy foggy day. Create a diffused lighting effect. Add mist on the scene."
cloudy_prompt = "Transform it into a heavy overcast day. Create a flat lighting effect. Remove shadows."
negative_prompt = "low quality, bad quality, worst quality, sketches, noisy, blurry, jpeg artifacts, deformed, fused objects, poorly drawn features, low definition"

# --- Image Generator Function ---
def create_variant(image: Image.Image, prompt: str, seed: int, guidance_scale: float, image_scale: float, steps: int) -> Image.Image:
    """Generates a single weather variant using the global pipe_edit."""
    # Each process will have its own generator, seeded appropriately
    current_process_generator = torch.Generator("cuda") # Use the current device's generator
    if seed != 0:
        current_process_generator.manual_seed(seed)
    else:
        current_process_generator.seed()

    new_width, new_height = image.size
    edited_image = pipe_edit(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            height=new_height,
            width=new_width,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_scale,
            num_inference_steps=steps,
            generator=current_process_generator, # Use process-specific generator
        ).images[0]
    return edited_image

# --- Copy File Structure and Non-Image Files for a specific weather type ---
def copy_directory_structure_for_weather(src_dir, dest_dir, weather_type):
    """Copy directory structure from src_dir to dest_dir/weather_type, excluding files"""
    print(f"Process {os.getpid()} ({weather_type}): Copying directory structure...")
    for dirpath, dirnames, _ in os.walk(src_dir):
        structure = os.path.relpath(dirpath, src_dir)
        if structure == ".": # Skip the root directory itself
            # Ensure the base weather directory exists
            os.makedirs(os.path.join(dest_dir, weather_type), exist_ok=True)
            continue
        new_dir = os.path.join(dest_dir, weather_type, structure)
        os.makedirs(new_dir, exist_ok=True)

def copy_non_image_files_for_weather(src_dir, dest_dir, weather_type):
    """Copy all non-image files to the specific weather variant directory"""
    non_image_files = []
    print(f"Process {os.getpid()} ({weather_type}): Finding non-image files...")
    for dirpath, _, filenames in os.walk(src_dir):
        for filename in filenames:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, src_dir)
                non_image_files.append(rel_path)

    if not non_image_files:
        print(f"Process {os.getpid()} ({weather_type}): No non-image files found to copy.")
        return

    print(f"Process {os.getpid()} ({weather_type}): Found {len(non_image_files)} non-image files. Copying for {weather_type}...")
    for file_path in tqdm(non_image_files, desc=f"Copying for {weather_type} (PID {os.getpid()})"):
        src_file_path = os.path.join(src_dir, file_path)
        dest_file_path = os.path.join(dest_dir, weather_type, file_path)
        os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
        try:
            shutil.copy2(src_file_path, dest_file_path)
        except Exception as e:
            print(f"Warning (PID {os.getpid()}): Could not copy {src_file_path} to {dest_file_path}: {e}")

# --- Image Processing Function ---
def process_image(args_tuple):
    image_path, src_dir, dest_dir, seed, guidance_scale, image_scale, steps, weather_to_process = args_tuple
    
    prompts_map = {
        "rainy": rainy_prompt,
        "night": night_prompt,
        "cloudy": cloudy_prompt,
        "foggy": foggy_prompt
    }
    current_prompt = prompts_map[weather_to_process]

    try:
        image_pil = load_image(image_path)
        image_pil_resized = resize_image(image_pil)

        if image_pil_resized.size[0] <= 0 or image_pil_resized.size[1] <= 0:
             print(f"Warning (PID {os.getpid()}): Image {image_path} resulted in invalid dimensions after resize. Skipping.")
             return

        rel_path = os.path.relpath(image_path, src_dir)
        
        noisy_image_pil = create_variant(
            image=image_pil_resized,
            prompt=current_prompt,
            seed=seed,
            guidance_scale=guidance_scale,
            image_scale=image_scale,
            steps=steps
        )

        target_path = os.path.join(dest_dir, weather_to_process, rel_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        noisy_image_pil.save(target_path)

    except FileNotFoundError:
        print(f"Warning (PID {os.getpid()}): Image file not found: {image_path}. Skipping.")
    except UnidentifiedImageError:
        print(f"Warning (PID {os.getpid()}): Could not identify or open image file: {image_path}. Skipping.")
    except Exception as e:
        print(f"Error (PID {os.getpid()}) processing image {image_path} for {weather_to_process}: {e}")

# --- Main Dataset Building Function (for a single weather type) ---
def build_noisy_dataset_for_weather(src_dir, dest_dir, seed, guidance_scale, image_scale, steps, weather_to_process):
    print(f"Process {os.getpid()}: Building '{weather_to_process}' variants from '{src_dir}' to '{os.path.join(dest_dir, weather_to_process)}'")
    
    # Ensure the main destination directory for this weather type exists
    weather_dest_dir = os.path.join(dest_dir, weather_to_process)
    os.makedirs(weather_dest_dir, exist_ok=True)

    copy_directory_structure_for_weather(src_dir, dest_dir, weather_to_process)
    copy_non_image_files_for_weather(src_dir, dest_dir, weather_to_process)

    print(f"Process {os.getpid()} ({weather_to_process}): Finding image files...")
    image_files = []
    patterns_to_try = [
        "**/image_[0-3][0-9]/data/*.png"
    ]
    for pattern in patterns_to_try:
        found = glob.glob(os.path.join(src_dir, pattern), recursive=True)
        if found:
            image_files.extend(found)
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print(f"\nError (PID {os.getpid()}): No image files found in {src_dir} for {weather_to_process}.")
        return

    print(f"\nProcess {os.getpid()} ({weather_to_process}): Found {len(image_files)} unique images. Preparing for processing.")
    process_args_list = [(img_path, src_dir, dest_dir, seed, guidance_scale, image_scale, steps, weather_to_process)
                         for img_path in image_files]

    print(f"Process {os.getpid()} ({weather_to_process}): Starting image processing...")
    for args_tuple in tqdm(process_args_list, total=len(image_files), desc=f"Processing images for {weather_to_process} (PID {os.getpid()})"):
        process_image(args_tuple)

    print(f"\nProcess {os.getpid()} ({weather_to_process}): Image processing finished.")

    marker_path = os.path.join(weather_dest_dir, f".generated_{weather_to_process}_v2_dist")
    try:
        with open(marker_path, "w") as f:
            device_name = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'CPU'
            f.write(f"This directory contains {weather_to_process} variants generated on {device_name} (PID {os.getpid()}).")
    except Exception as e:
        print(f"Warning (PID {os.getpid()}): Could not write marker file {marker_path}: {e}")

    print(f"Process {os.getpid()}: {weather_to_process} variants dataset build complete!")

# --- Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(description="Build specific weather variant of KITTI-like dataset (Distributed Task)")
    parser.add_argument("--src", required=True, help="Path to the original dataset directory")
    parser.add_argument("--dest", required=True, help="Base output path for weather variant datasets")
    parser.add_argument("--weather_type", required=True, choices=["rainy", "night", "cloudy", "foggy"], help="Specific weather type to generate for this task.")
    parser.add_argument("--seed", type=int, default=18457, help="Global seed for generation. Default: 18457")
    parser.add_argument("--guidance_scale", "--cfg", type=float, default=6.75, help="Guidance scale (CFG). Default: 6.75")
    parser.add_argument("--image_scale", "--img_cfg", type=float, default=2.5, help="Image guidance scale. Default: 2.5")
    parser.add_argument("--steps", type=int, default=100, help="Number of denoising steps. Default: 100")
    
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        print(f"Error (PID {os.getpid()}): Source directory not found: {args.src}")
        return
    # Destination directory will be created by build_noisy_dataset_for_weather if needed.
    # os.makedirs(args.dest, exist_ok=True) # Base dest dir

    if os.path.abspath(args.src) == os.path.abspath(args.dest):
        print(f"Error (PID {os.getpid()}): Source and base destination directories cannot be the same.")
        return

    build_noisy_dataset_for_weather(
        args.src,
        args.dest,
        args.seed,
        args.guidance_scale,
        args.image_scale,
        args.steps,
        args.weather_type
    )

if __name__ == "__main__":
    if not torch.cuda.is_available():
         print(f"FATAL (PID {os.getpid()}): CUDA is not available. Exiting.")
         exit(1)
    
    # Model loading and device check is now at the top global scope,
    # ensuring it happens after CUDA availability check and for each process.
    # The print statement in the global scope already confirms model device.
    print(f"Process {os.getpid()}: CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())} for weather type.")
    main()

