# -*- coding: utf-8 -*-
"""
Script for fine-tuning the Variational Autoencoder (VAE) component of a pre-trained Stable Diffusion model.

This script focuses solely on optimizing the VAE for better image reconstruction,
potentially improving the quality or style adaptation of generated images.
It utilizes the `diffusers` library for model handling, `accelerate` for distributed training
and mixed-precision support, and `Minio` for loading data from an S3-compatible object store.

Key functionalities:
- Loads a pre-trained Stable Diffusion model's VAE and UNet (UNet is frozen).
- Connects to a Minio instance to download training images.
- Defines a custom PyTorch Dataset (`OursDataset`) to load and preprocess images.
- Sets up training arguments using `argparse`.
- Configures the `accelerate` library for efficient training across multiple GPUs/TPUs.
- Implements a training loop that calculates the Mean Squared Error (MSE) reconstruction loss
  between the original image and the VAE's decoded output.
- Optimizes only the VAE parameters using AdamW optimizer and a configurable learning rate scheduler.
- Handles gradient accumulation, gradient checkpointing, and mixed-precision training.
- Saves training checkpoints periodically.
- Supports resuming training from checkpoints.
"""

import io
import pandas as pd
import numpy as np
import os
import cv2  # OpenCV for image loading and processing
from minio import Minio  # Client for S3-compatible object storage
from minio.error import S3Error  # Specific error for Minio operations
from functools import lru_cache  # For caching the Minio client instance

# --- Hugging Face and PyTorch specific imports ---
import argparse  # For parsing command-line arguments
import logging  # For logging information
import math  # For mathematical operations (e.g., ceil)
import random  # For setting random seeds
from pathlib import Path  # For handling file paths
from typing import Optional  # For type hinting

import accelerate  # Library for simplifying distributed training and mixed precision
import datasets  # Hugging Face datasets library
import torch  # PyTorch deep learning framework
import torch.nn.functional as F  # PyTorch functional API (e.g., for loss functions)
import torch.utils.checkpoint  # For gradient checkpointing to save memory
import transformers  # Hugging Face transformers library (used for CLIP model components)
from accelerate import Accelerator  # Core class from accelerate
from accelerate.logging import get_logger  # Customized logger from accelerate
from accelerate.utils import ProjectConfiguration, set_seed  # Utilities from accelerate
from datasets import load_dataset  # Function to load datasets from Hugging Face Hub or local files
from huggingface_hub import HfFolder, Repository, create_repo, whoami  # Utilities for interacting with Hugging Face Hub
from packaging import version  # For comparing package versions
from torchvision import transforms  # Common image transformations
from tqdm.auto import tqdm  # Progress bar utility
from transformers import CLIPTextModel, CLIPTokenizer  # CLIP components (loaded but not trained/used in loss)
from torch.utils.data import Dataset  # Base class for PyTorch datasets
import diffusers  # Hugging Face library for diffusion models
from PIL import Image, ImageDraw  # Python Imaging Library for image manipulation
import json  # For handling JSON data (e.g., metadata)
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, \
    UNet2DConditionModel  # Core components of Diffusion models
from diffusers.optimization import get_scheduler  # Learning rate schedulers
from diffusers.training_utils import EMAModel  # Exponential Moving Average model utility (optional)
from diffusers.utils import check_min_version, deprecate  # Utilities from diffusers
from diffusers.utils.import_utils import is_xformers_available  # Check for xFormers library for optimized attention

# --- Specific configurations and patches ---
from alps.pytorch.api.utils.web_access import \
    patch_requests  # Potentially internal library for patching web requests (specific to ALPS environment?)
import cv2
import albumentations as alb  # Library for image augmentations
from albumentations.pytorch import ToTensorV2  # Albumentations transform to convert to PyTorch tensor

patch_requests()  # Apply the web request patch

import torch.multiprocessing

# Set the multiprocessing sharing strategy to 'file_system'
# This can help avoid "Too many open files" errors in certain environments when using many data workers.
torch.multiprocessing.set_sharing_strategy('file_system')

# Ensure a minimum version of diffusers is installed, critical for compatibility.
check_min_version("0.15.0.dev0")

# Set CUDA environment variable for debugging asynchronous kernel launches.
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Get a logger instance configured by accelerate.
logger = get_logger(__name__, log_level="INFO")

# --- MinIO Configuration ---
# Load MinIO connection details from environment variables with defaults.
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "play.min.io")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "your-access-key")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "your-secret-key")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "your-bucket")
MINIO_SECURE = os.getenv("MINIO_SECURE", "True").lower() == "true"


@lru_cache(maxsize=1)
def get_minio_client() -> Minio:
    """
    Creates and returns a singleton MinIO client instance.

    Uses `lru_cache` to ensure only one client object is created,
    improving efficiency and resource management.

    Returns:
        Minio: An initialized MinIO client object.

    Raises:
        Exception: If the client fails to initialize (e.g., invalid credentials).
    """
    try:
        logger.info(f"Initializing MinIO client for endpoint: {MINIO_ENDPOINT}, bucket: {MINIO_BUCKET}")
        client = Minio(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        # Optionally, add a check here to ensure the bucket exists
        # found = client.bucket_exists(MINIO_BUCKET)
        # if not found:
        #     logger.error(f"MinIO bucket '{MINIO_BUCKET}' not found.")
        #     raise ValueError(f"MinIO bucket '{MINIO_BUCKET}' not found.")
        return client
    except Exception as e:
        logger.error(f"Failed to create MinIO client: {str(e)}")
        raise


def download_file_minio(file_path: str) -> np.ndarray:
    """
    Downloads a file (expected to be an image) from the configured MinIO bucket
    and decodes it into a NumPy array (OpenCV format BGR).

    Args:
        file_path (str): The path (key) of the file within the MinIO bucket.

    Returns:
        np.ndarray: The decoded image as a NumPy array (BGR format).

    Raises:
        S3Error: If a MinIO-specific error occurs during download.
        Exception: For any other errors during download or image decoding.
    """
    try:
        client = get_minio_client()
        logger.debug(f"Attempting to download from MinIO: bucket='{MINIO_BUCKET}', path='{file_path}'")

        # Get object data stream from MinIO
        data = client.get_object(MINIO_BUCKET, file_path)

        # Read the data stream into an in-memory bytes buffer
        buffer = io.BytesIO()
        for d in data.stream(32 * 1024):  # Read in 32KB chunks
            buffer.write(d)
        buffer.seek(0)  # Reset buffer position to the beginning

        # Convert raw bytes to NumPy array
        file_bytes = np.frombuffer(buffer.read(), dtype=np.uint8)
        # Decode the byte array into an image using OpenCV (loads as BGR by default)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # flags=1 is equivalent to IMREAD_COLOR

        if img is None:
            raise ValueError(f"Failed to decode image from MinIO path: {file_path}")

        logger.debug(f"Successfully downloaded and decoded image from MinIO: {file_path}")
        return img
    except S3Error as e:
        logger.error(f"MinIO S3 error while downloading {file_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error processing file {file_path} from MinIO: {str(e)}")
        raise


# Clear PyTorch CUDA cache to free up GPU memory before starting.
torch.cuda.empty_cache()


def parse_args():
    """
    Parses command-line arguments for the training script.
    """
    parser = argparse.ArgumentParser(description="Script for fine-tuning the VAE of a Stable Diffusion model.")

    # --- Model and Data Paths ---
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to the pretrained Stable Diffusion model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Specific model version (e.g., branch, tag, commit hash) to use.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the dataset on the HuggingFace Hub or a local path to a dataset."
            " (Note: This script currently uses a hardcoded CSV and MinIO, this arg might be unused)."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config name of the dataset (if applicable).",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing training data. (Note: This script currently uses MinIO, this arg might be unused)."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image",
        help="Column name for images in the dataset (if using datasets library)."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="Column name for captions (unused in this VAE-only training script).",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training examples to use (for debugging/quick runs).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-vae-finetuned",  # Changed default name to reflect VAE focus
        help="Directory where model checkpoints and logs will be saved.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for caching downloaded models and datasets.",
    )

    # --- Training Hyperparameters ---
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Input image resolution. Images will be resized/cropped to this size.",
    )
    # Guidance scale is typically for inference, might be irrelevant for VAE training loss, but kept for potential compatibility/future use.
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale (often used in inference, potentially unused here).")  # Defaulted to common SD value
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="Whether to center crop images instead of random cropping.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="Whether to randomly flip images horizontally (applied via albumentations).",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size per GPU/TPU device."
        # Reduced default for VAE training memory
    )
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Total number of training epochs.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps. Overrides num_train_epochs if set.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before performing an optimizer update.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory (at the cost of slower training).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,  # Often lower LR for fine-tuning
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale learning rate based on number of GPUs, batch size, and accumulation steps.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'Learning rate scheduler type. Options: ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of warmup steps for the learning rate scheduler."
        # Defaulted to 0 for constant scheduler
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Use 8-bit Adam optimizer (requires bitsandbytes).",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Allow TF32 precision on Ampere GPUs for potentially faster training.",
    )
    parser.add_argument("--use_ema", action="store_true",
                        help="Use Exponential Moving Average (EMA) for model weights (currently applies to UNet if enabled, might need adjustment for VAE).")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of the non-EMA model weights (if applicable and different from main revision)."
            " Use `--variant=non_ema` with newer diffusers versions."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,  # Adjusted default
        help="Number of subprocesses for data loading.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam optimizer beta1 parameter.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam optimizer beta2 parameter.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay for Adam optimizer.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Adam optimizer epsilon parameter.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Maximum gradient norm for clipping.")

    # --- Logging, Checkpointing, and Hub ---
    parser.add_argument("--push_to_hub", action="store_true", help="Push the final model to the Hugging Face Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="Hugging Face Hub authentication token.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Repository name on the Hub to push to (e.g., your-username/sd-vae-finetuned).",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory for TensorBoard logs (relative to output_dir).",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,  # Will default to accelerate config
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training type ('fp16', 'bf16', or 'no').",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='Log reporting integration(s). Options: "tensorboard", "wandb", "comet_ml", "all".',
    )
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (set automatically by accelerate).")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a training checkpoint every X steps.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,  # Keep all checkpoints by default
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Resume training from a specific checkpoint directory or "latest".',
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true",
        help="Use xFormers for memory-efficient attention in UNet (if available)."
    )

    args = parser.parse_args()

    # --- Post-processing and Sanity Checks ---
    # Set local rank from environment variable if available, overriding the argument.
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        logger.info(f"Overriding local_rank {args.local_rank} with environment variable LOCAL_RANK {env_local_rank}")
        args.local_rank = env_local_rank

    # Ensure either a dataset name or data directory is provided (though this script overrides with MinIO/CSV).
    # if args.dataset_name is None and args.train_data_dir is None:
    #     # Adjusted check because this script uses a hardcoded CSV path
    #     if not os.path.exists('data.csv'):
    #          raise ValueError("Need 'data.csv' in the current directory or specify --dataset_name or --train_data_dir (though these are currently overridden).")
    # For now, we assume data.csv exists and MinIO is configured.

    # Default to using the same revision for non-EMA weights if not specified.
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    # Ensure output directory exists if not pushing to hub
    if not args.push_to_hub and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


# --- Helper Functions (Potentially unused in the current VAE training loop) ---

def generate_mask(im_shape: tuple, ocr_locate: list) -> Image.Image:
    """
    Generates a binary mask image with a white rectangle at the specified location.
    (Likely intended for inpainting tasks, but not used in the main VAE training loop).

    Args:
        im_shape (tuple): The (width, height) of the mask image to create.
        ocr_locate (list): A list representing the rectangle coordinates [x1, y1, x2, y2].

    Returns:
        Image.Image: A PIL Image object representing the binary mask (mode "L").
    """
    mask = Image.new("L", im_shape, 0)  # Create a black image
    draw = ImageDraw.Draw(mask)
    # Draw a white rectangle based on the coordinates
    draw.rectangle(
        (ocr_locate[0], ocr_locate[1], ocr_locate[2], ocr_locate[3]),
        fill=255,  # White
    )
    return mask


def prepare_mask_and_masked_image(image: Image.Image, mask: Image.Image) -> tuple:
    """
    Prepares mask and masked image tensors for inpainting models.
    Converts PIL images to PyTorch tensors, normalizes them, and applies the mask.
    (Likely intended for inpainting tasks, but not used in the main VAE training loop).

    Args:
        image (Image.Image): The original image (RGB).
        mask (Image.Image): The mask image (L).

    Returns:
        tuple: A tuple containing:
            - mask (torch.Tensor): The mask tensor (1, 1, H, W), values are 0 or 1.
            - masked_image (torch.Tensor): The image tensor with masked areas zeroed out (1, 3, H, W), normalized to [-1, 1].
    """
    # Convert image to numpy array, add batch dimension, permute to (B, C, H, W)
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    # Convert to float tensor and normalize to [-1, 1]
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    # Convert mask to numpy array (L mode), normalize to [0, 1]
    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    # Add batch and channel dimensions
    mask = mask[None, None]
    # Binarize the mask (ensure values are exactly 0 or 1)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    # Apply mask: image areas where mask is 1 (masked region) become 0
    masked_image = image * (mask < 0.5)  # Invert mask logic here if needed (mask=1 means keep, mask=0 means erase)

    return mask, masked_image


# NOTE: This function seems intended for a specific environment (OSS_PCACHE_ROOT_DIR)
# and uses a different file I/O library (`fileio`). It conflicts with the `download_file_minio`
# function defined earlier. The dataset code calls this, which might be an error.
# Assuming the intent was to use the MinIO downloader.
# def download_oss_file_pcache(my_file = "xxx"):
#     """
#     Downloads a file from a specific path likely related to an OSS Pcache system.
#     This function seems specific to a particular infrastructure and might not be generally applicable.
#     It conflicts with the `download_file_minio` function.
#     """
#     OSS_PCACHE_ROOT_DIR = os.getenv("OSS_PCACHE_ROOT_DIR", "/cache/path/prefix") # Example placeholder
#     MY_FILE_PATH = os.path.join(OSS_PCACHE_ROOT_DIR, my_file)
#     try:
#         # This part uses a specific 'fileio' library, likely internal or specific to the environment.
#         with fileio.file_io_impl.open(MY_FILE_PATH, "rb") as fd:
#             content = fd.read()
#         img = np.frombuffer(content, dtype=np.int8) # Note: dtype might need to be uint8
#         img = cv2.imdecode(img, flags=1) # flags=1 means cv2.IMREAD_COLOR
#         if img is None:
#             raise ValueError(f"Failed to decode image from pcache path: {MY_FILE_PATH}")
#         return img
#     except Exception as e:
#         logger.error(f"Error downloading/decoding from pcache {MY_FILE_PATH}: {e}")
#         # Return a placeholder or re-raise depending on desired behavior
#         # Example: return np.zeros((512, 512, 3), dtype=np.uint8)
#         raise


# --- Albumentations Image Transformations ---
# Define image transformations using Albumentations library.

# 1. Resize and Crop: Applied first to get the target resolution.
#    - Randomly crops a 512x512 patch. If the image is smaller, this might error or behave unexpectedly.
#      Consider adding a SmallestMaxSize or LongestMaxSize before RandomCrop if input sizes vary significantly below 512.
#    - Resizes the crop to 512x512 (might be redundant if RandomCrop is already 512x512).
#    - Normalizes the image pixel values to the range [-1, 1], suitable for VAE input.
image_trans_resize_and_crop = alb.Compose(
    [  # Consider adding alb.SmallestMaxSize(max_size=512) or similar before crop if needed
        alb.RandomCrop(height=512, width=512, p=1.0),
        alb.Resize(height=512, width=512),  # Ensure final size is 512x512
        alb.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ])

# 2. Final Conversion: Applied after resize/crop and normalization.
#    - Converts the NumPy array (output of Albumentations) to a PyTorch tensor.
#      The channels dimension will be placed first (C, H, W), as expected by PyTorch models.
image_trans = alb.Compose(
    [ToTensorV2(), ])  # Converts HWC NumPy array to CHW PyTorch Tensor


# --- Custom PyTorch Dataset ---
class OursDataset(Dataset):
    """
    Custom PyTorch Dataset for training the VAE component of DiffUTE.
    
    This dataset class handles loading and preprocessing of images for VAE training.
    It supports loading images from either local storage or MinIO cloud storage,
    and applies necessary transformations to prepare the images for training.

    Attributes:
        size (int): Target size for image resizing (default: 512).
        center_crop (bool): Whether to apply center cropping (currently unused).
        transform_resize_crop (alb.Compose): Albumentations transforms for resizing/cropping.
        transform (alb.Compose): Additional Albumentations transforms.
        image_paths (list): List of paths to all images in the dataset.
    """

    def __init__(
            self,
            size=512,
            center_crop=False,  # Note: This argument is not currently used by the hardcoded transforms
            transform_resize_crop=None,
            transform=None,
    ):
        """
        Initialize the dataset with specified parameters and transformations.

        Args:
            size (int): Target size for image resizing.
            center_crop (bool): Flag for center cropping (currently unused).
            transform_resize_crop (alb.Compose, optional): Albumentations transforms for resizing/cropping.
            transform (alb.Compose, optional): Additional Albumentations transforms.
        """
        self.size = size
        self.center_crop = center_crop
        self.transform_resize_crop = transform_resize_crop
        self.transform = transform

        # Load all image paths during initialization
        self.image_paths = self._load_images_paths()

        # Set up default transforms if none provided
        if self.transform_resize_crop is None:
            self.transform_resize_crop = alb.Compose([
                alb.SmallestMaxSize(max_size=size),
                alb.CenterCrop(height=size, width=size),
            ])

        if self.transform is None:
            self.transform = alb.Compose([
                alb.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ])

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def _load_images_paths(self):
        """Loads image file paths from the 'data.csv' file."""
        csv_path = 'data.csv'
        try:
            logger.info(f"Loading training image paths from {csv_path}...")
            df = pd.read_csv(csv_path, low_memory=False)
            # Expecting a column named 'path' containing relative paths for MinIO
            if 'path' not in df.columns:
                raise ValueError("CSV file 'data.csv' must contain a 'path' column.")
            # Prepend 'data/' prefix as seems intended by original code snippet
            # Make sure this prefix matches the structure in your MinIO bucket
            # path_prefix = 'data/'
            # self.instance_paths = (path_prefix + df['path']).tolist()
            # OR if the 'path' column already contains the full path needed for MinIO:
            self.instance_paths = df['path'].tolist()
            logger.info(f"Found {len(self.instance_paths)} paths in {csv_path}.")
        except FileNotFoundError:
            logger.error(f"Error: {csv_path} not found. Please ensure it exists in the working directory.")
            raise
        except Exception as e:
            logger.error(f"Error loading or processing {csv_path}: {e}")
            raise

    def __getitem__(self, index):
        """
        Get a single item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'pixel_values': Preprocessed image tensor normalized to [-1, 1]
                - 'index': Original index in the dataset

        Raises:
            Exception: If image loading or processing fails.
        """
        example = {}
        # Use modulo operator for safety, although index should be within bounds
        image_path = self.instance_paths[index % self.num_instance_images]

        try:
            # --- Download Image ---
            # Use the MinIO download function defined earlier
            instance_image = download_file_minio(image_path)
            # Original code called 'download_oss_file_pcache', which seems incorrect/environment-specific.
            # instance_image = download_oss_file_pcache(image_path) # This line is likely wrong

            if instance_image is None:
                raise ValueError("Downloaded image is None")

            # Convert BGR (OpenCV default) to RGB (Albumentations/PIL default)
            instance_image = cv2.cvtColor(instance_image, cv2.COLOR_BGR2RGB)

            # --- Pre-resize (Optional but Recommended) ---
            # Upscale small images before random cropping to avoid low detail crops.
            h, w, c = instance_image.shape
            min_side = min(h, w)
            target_size = self.size  # e.g., 512
            # If the smallest side is less than our target crop size, upscale it.
            # Using a larger intermediate size like 1024 might preserve more detail if original is very small.
            # intermediate_size = max(target_size, 768) # Example intermediate size
            intermediate_size = target_size  # Or just upscale to target size directly
            if min_side < intermediate_size:
                scale_factor = intermediate_size / min_side
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                # Use INTER_AREA for shrinking, INTER_LANCZOS4 or INTER_CUBIC for enlarging
                interpolation = cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LANCZOS4
                instance_image = cv2.resize(instance_image, (new_w, new_h), interpolation=interpolation)
                logger.debug(f"Resized image from {(h, w)} to {(new_h, new_w)} before cropping.")

            # --- Apply Transformations ---
            # Apply resize/crop and normalization transforms
            if self.transform_resize_crop:
                augmented = self.transform_resize_crop(image=instance_image)
                instance_image = augmented["image"]  # Now normalized numpy array (H, W, C)
            else:
                # If no resize/crop transform, ensure image is at least size x size
                # This part might need more robust handling depending on expected inputs
                h, w, _ = instance_image.shape
                if h < self.size or w < self.size:
                    # Basic resize if too small, might distort aspect ratio
                    instance_image = cv2.resize(instance_image, (self.size, self.size),
                                                interpolation=cv2.INTER_LANCZOS4)
                # Apply normalization manually if needed
                # instance_image = (instance_image / 255.0) * 2.0 - 1.0 # Normalize [0, 255] -> [-1, 1]

            # Apply final transform (ToTensorV2)
            if self.transform:
                # ToTensorV2 expects HWC, which is the output format of albumentations transforms
                augmented = self.transform(image=instance_image)
                instance_image = augmented["image"]  # Now a CHW PyTorch tensor
            else:
                # Convert manually if transform is missing
                instance_image = torch.from_numpy(instance_image).permute(2, 0, 1).float()

            example["instance_images"] = instance_image
            return example

        except Exception as e:
            logger.error(f"Error processing image at index {index} (path: {image_path}): {e}")
            # Return a dummy/empty example or handle differently
            # Returning an empty dict might cause issues in collation, consider skipping or returning placeholder
            # For simplicity, re-raising might be better during debugging
            raise e
            # return {} # Or return None and handle in collate_fn


# --- Hugging Face Hub Helper Function ---
def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    """
    Constructs the full repository name for the Hugging Face Hub.

    Args:
        model_id (str): The base name of the repository (e.g., "sd-vae-finetuned").
        organization (Optional[str]): The organization name. If None, uses the logged-in user's username.
        token (Optional[str]): Hugging Face Hub authentication token. Uses cached token if None.

    Returns:
        str: The full repository name (e.g., "username/sd-vae-finetuned" or "org/sd-vae-finetuned").
    """
    if token is None:
        token = HfFolder.get_token()  # Get cached token
    if organization is None:
        try:
            username = whoami(token)["name"]  # Get username associated with the token
            return f"{username}/{model_id}"
        except Exception as e:
            logger.error(
                f"Could not get username from Hugging Face Hub token: {e}. Please ensure you are logged in (`huggingface-cli login`) or provide organization.")
            raise e  # Or handle more gracefully depending on requirements
    else:
        return f"{organization}/{model_id}"


# --- Tensor to Image Conversion Helper (Potentially unused) ---
def tensor2im(input_image, imtype=np.uint8) -> np.ndarray:
    """
    Converts a PyTorch tensor representing an image to a NumPy array for visualization.
    Assumes the tensor is normalized in the range [-1, 1].

    (Not used in the main training loop but can be useful for debugging/visualization).

    Args:
        input_image (torch.Tensor or np.ndarray): Input image tensor (typically CHW or BCHW format) or existing NumPy array.
        imtype (type): Desired NumPy data type for the output array (e.g., np.uint8 for display).

    Returns:
        np.ndarray: Image as a NumPy array in HWC format, scaled to [0, 255].
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            # Input is not a Tensor or ndarray, return as is
            return input_image

        # Move tensor to CPU, select the first image if batch, convert to numpy
        image_numpy = image_tensor[0].cpu().float().numpy()

        # Handle single-channel (grayscale) images by tiling
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        # Transpose from CHW to HWC and denormalize from [-1, 1] to [0, 255]
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        # If it's already a NumPy array, assume it's correctly formatted HWC
        image_numpy = input_image

    # Clip values to [0, 255] just in case and convert to the desired type
    return np.clip(image_numpy, 0, 255).astype(imtype)


# ==========================================
#               Main Function
# ==========================================
def main():
    """
    Main function to orchestrate the VAE fine-tuning process.
    """
    # 1. Parse Command Line Arguments
    args = parse_args()

    # Handle deprecated argument `non_ema_revision`
    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )

    # 2. Setup Logging and Accelerate
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    # Configure logging levels based on process rank
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # Base level
    )
    logger.info(accelerator.state, main_process_only=False)  # Log accelerator state on all processes

    if accelerator.is_local_main_process:
        # Set higher verbosity for libraries on the main process
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # Set lower verbosity on other processes
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 3. Set Random Seed (if provided)
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Set random seed to {args.seed}")

    # 4. Handle Repository Creation (Hugging Face Hub)
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                # Automatically generate repo name if not provided
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            logger.info(f"Creating or using Hub repository: {repo_name}")
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            # Initialize a local git repository linked to the Hub repo
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            # Add patterns to .gitignore to avoid uploading intermediate checkpoints
            with open(os.path.join(args.output_dir, ".gitignore"), "a+") as gitignore:  # Use append mode 'a+'
                gitignore.seek(0)  # Go to the beginning to read
                content = gitignore.read()
                if "step_*" not in content:
                    gitignore.write("step_*\n")
                if "epoch_*" not in content:
                    gitignore.write("epoch_*\n")

        elif args.output_dir is not None:
            # Ensure local output directory exists
            os.makedirs(args.output_dir, exist_ok=True)
            logger.info(f"Output directory set to: {args.output_dir}")

    # 5. Load Models (VAE and UNet)
    # Note: Tokenizer and Text Encoder are not loaded as they are not needed for VAE training.
    logger.info(f"Loading VAE from: {args.pretrained_model_name_or_path}")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        cache_dir=args.cache_dir,
    )

    logger.info(f"Loading UNet from: {args.pretrained_model_name_or_path} (will be frozen)")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.non_ema_revision,  # Use non_ema_revision if specified
        cache_dir=args.cache_dir,
    )

    # --- Freeze Models ---
    # Freeze the UNet parameters, as we are only training the VAE.
    unet.requires_grad_(False)
    logger.info("UNet parameters frozen.")
    # VAE parameters remain trainable by default.

    # 6. Optional: Enable xFormers Memory Efficient Attention (for UNet, if used later)
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            try:
                unet.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xFormers memory efficient attention for UNet.")
            except Exception as e:
                logger.warning(f"Could not enable xFormers memory efficient attention: {e}. Proceeding without it.")
        else:
            logger.warning("xFormers is not available. Install it for potential memory savings.")
            # raise ValueError("xformers is not available. Make sure it is installed correctly")

    # 7. Optional: Gradient Checkpointing (for VAE)
    # Gradient checkpointing saves memory during backpropagation by recomputing intermediate activations.
    # This is applied to the VAE here, as it's the model being trained.
    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()
        logger.info("Enabled gradient checkpointing for VAE.")

    # 8. Optional: Allow TF32 Precision (for Ampere+ GPUs)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.info("Enabled TF32 precision for matmul operations.")

    # 9. Scale Learning Rate (if requested)
    if args.scale_lr:
        initial_lr = args.learning_rate
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        logger.info(f"Scaled learning rate from {initial_lr} to {args.learning_rate}")

    # 10. Initialize Optimizer
    optimizer_cls = None
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            logger.info("Using 8-bit AdamW optimizer.")
        except ImportError:
            logger.error(
                "bitsandbytes library not found, but --use_8bit_adam was specified. Falling back to regular AdamW."
                " Install bitsandbytes (`pip install bitsandbytes`) to use 8-bit Adam."
            )
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW
        logger.info("Using standard AdamW optimizer.")

    optimizer = optimizer_cls(
        vae.parameters(),  # Pass only VAE parameters to the optimizer
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 11. Prepare Dataset and DataLoader
    logger.info("Initializing dataset...")
    train_dataset = OursDataset(
        size=args.resolution,
        center_crop=args.center_crop,  # Pass argument, although transform might override
        transform_resize_crop=image_trans_resize_and_crop,  # Pass the defined transforms
        transform=image_trans
    )

    # Collate function remains simple as dataset returns a dict with a single tensor
    def collate_fn_simple(examples):
        """Basic collate function assuming dataset returns dict with 'instance_images'."""
        # Filter out potential problematic examples (e.g., if __getitem__ returned None or empty dict)
        # valid_examples = [ex for ex in examples if ex and "instance_images" in ex]
        # if not valid_examples:
        #     # Handle case where the entire batch is invalid
        #     logger.warning("Collate function received an empty or invalid batch.")
        #     # Returning empty tensors might be necessary for accelerator compatibility
        #     return {"pixel_values": torch.Tensor()}
        #     # Or raise an error: raise ValueError("Batch contains no valid examples")

        # If not filtering, assume all examples are valid
        pixel_values = torch.stack([example["instance_images"] for example in examples])
        # Ensure tensor is in contiguous memory format and correct dtype
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values}

    logger.info("Initializing DataLoader...")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,  # Shuffle data for training
        collate_fn=collate_fn_simple,  # Use the simple collate function
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,  # Helps speed up data transfer to GPU
        drop_last=True  # Drop last incomplete batch
    )

    # 12. Setup Learning Rate Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        # Calculate max_train_steps based on epochs if not provided
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        logger.info(f"Calculated max_train_steps: {args.max_train_steps} ({args.num_train_epochs} epochs)")
    else:
        # If max_train_steps is provided, recalculate num_train_epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        overrode_max_train_steps = False
        logger.info(
            f"Using provided max_train_steps: {args.max_train_steps}. Adjusted num_train_epochs: {args.num_train_epochs}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,  # Scale warmup steps by accumulation
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,  # Scale total steps by accumulation
    )
    logger.info(f"Using learning rate scheduler: {args.lr_scheduler} with {args.lr_warmup_steps} warmup steps.")

    # 13. Prepare Components with Accelerator
    logger.info("Preparing models, optimizer, dataloader, and scheduler with accelerate...")
    # Pass only the components that need preparation (models being trained/evaluated, optimizer, dataloader, scheduler)
    vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_dataloader, lr_scheduler
    )
    # Note: UNet is NOT passed to accelerator.prepare as it's frozen and not part of the optimization loop.
    # It should be moved to the correct device manually if needed later (e.g., for evaluation)
    unet = unet.to(accelerator.device)

    # Define the weight dtype based on mixed precision setting
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        logger.info("Using fp16 mixed precision.")
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        logger.info("Using bf16 mixed precision.")
    else:
        logger.info("Using fp32 precision.")

    # Move models to the correct device and dtype
    # VAE is already handled by accelerator.prepare
    unet.to(accelerator.device, dtype=weight_dtype)  # Move frozen UNet to device and potentially cast dtype
    # VAE's dtype is managed by accelerator based on mixed_precision

    # Recalculate training steps/epochs after dataloader preparation (in case it changed size due to distribution)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:  # If we calculated max_steps based on initial epoch count
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Always recalculate epochs based on the final max_steps and dataloader size
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 14. Initialize Trackers (TensorBoard, Wandb, etc.)
    if accelerator.is_main_process:
        # Initialize trackers only on the main process
        # Use a descriptive run name including relevant args
        run_name = f"vae_finetune_{Path(args.pretrained_model_name_or_path).name}_lr{args.learning_rate}_bs{args.train_batch_size}"
        accelerator.init_trackers(run_name, config=vars(args))
        logger.info(f"Initialized trackers ({args.report_to}) with run name: {run_name}")

    # 15. Training Loop Setup
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Starting VAE Fine-tuning *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, dist & accum) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # 16. Resume from Checkpoint (if applicable)
    if args.resume_from_checkpoint:
        resume_path = None
        if args.resume_from_checkpoint != "latest":
            resume_path = args.resume_from_checkpoint
            logger.info(f"Attempting to resume from specific checkpoint: {resume_path}")
        else:
            # Find the most recent checkpoint directory
            checkpoint_dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
            if checkpoint_dirs:
                checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
                resume_path = os.path.join(args.output_dir, checkpoint_dirs[-1])
                logger.info(f"Found latest checkpoint: {resume_path}")
            else:
                logger.warning(f"Resume from 'latest' requested, but no checkpoint found in {args.output_dir}.")

        if resume_path and os.path.isdir(resume_path):
            try:
                logger.info(f"Resuming training from checkpoint: {resume_path}")
                accelerator.load_state(resume_path)
                # Extract the global step number from the checkpoint directory name
                global_step = int(resume_path.split("-")[-1])
                logger.info(f"Resumed from global step: {global_step}")

                # Calculate starting epoch and step within the epoch
                resume_global_step = global_step * args.gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
                logger.info(f"Resuming at Epoch {first_epoch}, Step {resume_step} (in dataloader).")

            except Exception as e:
                logger.error(f"Failed to load checkpoint from {resume_path}: {e}. Starting training from scratch.")
                args.resume_from_checkpoint = None  # Reset flag if loading failed
                global_step = 0
                first_epoch = 0
                resume_step = 0  # Need to reset resume_step as well
        else:
            logger.warning(
                f"Checkpoint '{args.resume_from_checkpoint}' not found or not a directory. Starting training from scratch.")
            args.resume_from_checkpoint = None  # Reset flag if path invalid

    # 17. Initialize Progress Bar
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        initial=global_step,  # Start progress bar from the resumed step
                        disable=not accelerator.is_local_main_process,  # Show only on main process
                        desc="VAE Training Steps")

    # ==============================
    #       >>> Training Loop <<<
    # ==============================
    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{args.num_train_epochs}")
        vae.train()  # Set VAE to training mode
        train_loss = 0.0  # Accumulate loss over logging steps

        for step, batch in enumerate(train_dataloader):
            # Skip steps if resuming within an epoch
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                # Update progress bar even for skipped steps if they correspond to optimizer steps
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue  # Skip this batch

            # Perform training step under accelerator's context manager
            # This handles gradient accumulation and synchronization.
            with accelerator.accumulate(vae):
                # --- Forward Pass ---
                # Get input images, move to correct device and cast to appropriate dtype for the VAE
                # Input images are already normalized in the dataset's transform
                input_img = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)

                # Pass input images through the VAE to get reconstructions
                # The VAE output is a DiagonalGaussianDistribution object. Access the reconstructed sample via `.sample`
                reconstruction = vae(input_img).sample

                # --- Calculate Loss ---
                # Use Mean Squared Error (MSE) between original input and reconstruction
                # Ensure both tensors are float32 for loss calculation if using mixed precision
                # to avoid potential numerical instability or dtype mismatch issues.
                loss = F.mse_loss(reconstruction.float(), input_img.float(), reduction="mean")

                # --- Logging Loss ---
                # Gather loss across all devices for accurate average logging
                # Average loss = sum of losses / total batch size (across all devices)
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                # Accumulate loss for logging period (divide by grad accum steps for avg loss per step)
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # --- Backward Pass & Optimization ---
                accelerator.backward(loss)  # Perform backpropagation

                # Clip gradients if enabled and if gradients are synchronized
                if accelerator.sync_gradients:
                    # Only clip gradients when performing an optimizer step (after accumulation)
                    accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)

                optimizer.step()  # Update VAE weights
                lr_scheduler.step()  # Update learning rate
                optimizer.zero_grad()  # Clear gradients for the next accumulation cycle

            # --- Post-Optimization Step ---
            # Check if an optimizer step was performed (gradients were synchronized)
            if accelerator.sync_gradients:
                progress_bar.update(1)  # Increment progress bar by one optimizer step
                global_step += 1

                # Log metrics (average loss over the accumulation steps)
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0  # Reset accumulated loss

                # Update progress bar postfix with current step loss and LR
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                # --- Checkpointing ---
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # accelerator.save_state handles saving models, optimizer, scheduler, etc.
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")
                        # Optionally, push checkpoint to Hub if configured
                        # if args.push_to_hub and repo:
                        #     try:
                        #         logger.info(f"Pushing checkpoint {global_step} to Hub...")
                        #         repo.push_to_hub(commit_message=f"Training checkpoint step {global_step}", blocking=False) # Non-blocking push
                        #     except Exception as e:
                        #         logger.error(f"Failed to push checkpoint to Hub: {e}")

            # Check if max training steps reached
            if global_step >= args.max_train_steps:
                logger.info("Maximum training steps reached.")
                break  # Exit inner loop (step loop)

        # Check again if max training steps reached after finishing an epoch
        if global_step >= args.max_train_steps:
            break  # Exit outer loop (epoch loop)

    # 18. End of Training
    logger.info("Training finished.")
    accelerator.wait_for_everyone()  # Ensure all processes finish tasks

    # 19. Save Final Model
    if accelerator.is_main_process:
        # Unwrap the model if necessary (if prepared by accelerator)
        vae_final = accelerator.unwrap_model(vae)
        # Save the final VAE model in diffusers format
        final_save_path = os.path.join(args.output_dir, "final_vae")
        vae_final.save_pretrained(final_save_path)
        logger.info(f"Saved final VAE model to {final_save_path}")

        # Optionally, save as a complete pipeline (if desired, though only VAE was trained)
        # Requires loading tokenizer, text_encoder etc. if they weren't loaded before
        # pipeline = StableDiffusionPipeline.from_pretrained(
        #    args.pretrained_model_name_or_path,
        #    vae=vae_final, # Inject the fine-tuned VAE
        #    unet=accelerator.unwrap_model(unet), # Use the original UNet
        #    revision=args.revision,
        #    torch_dtype=weight_dtype, # Save in appropriate dtype
        # )
        # pipeline_save_path = os.path.join(args.output_dir, "final_pipeline")
        # pipeline.save_pretrained(pipeline_save_path)
        # logger.info(f"Saved final pipeline with tuned VAE to {pipeline_save_path}")

        # --- Push to Hub ---
        if args.push_to_hub and repo:
            try:
                # Use repo object initialized earlier
                logger.info(f"Pushing final model to Hub repository: {repo.repo_id}")
                # You might want to upload specific files (like the VAE folder) or the whole output dir
                repo.git_add(auto_lfs_track=True)  # Track large files with LFS
                repo.git_commit(f"End of VAE training epoch {epoch + 1}, global step {global_step}")
                repo.git_push()  # Push changes
                # Alternatively, use repo.push_to_hub() which combines add, commit, push
                # repo.push_to_hub(commit_message="End of VAE training", blocking=True) # Blocking push for final model
                logger.info("Successfully pushed final model to Hub.")
            except Exception as e:
                logger.error(f"Failed to push final model to Hub: {e}")

    # 20. Clean Up
    accelerator.end_training()
    logger.info("Accelerator training ended.")


if __name__ == "__main__":
    main()