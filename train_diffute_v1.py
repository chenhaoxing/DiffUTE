# -*- coding: utf-8 -*-
"""
This script fine-tunes a Stable Diffusion model for an image inpainting task,
specifically tailored for document images. It uses text information extracted
via OCR (Optical Character Recognition) as conditioning for the diffusion model.

The workflow involves:
1. Loading image data and corresponding OCR results (potentially from cloud storage like MinIO or OSS).
2. Processing OCR data to identify text regions (bounding boxes) and content.
3. Generating masks based on selected text regions.
4. Preparing masked images, original images, and text renderings.
5. Using a pre-trained TrOCR model to extract features from the rendered text, serving as conditioning.
6. Fine-tuning the UNet part of a Stable Diffusion pipeline using these inputs (masked image latent, mask, original image latent, text conditioning).
7. Leveraging Hugging Face's Accelerate library for efficient distributed training and mixed precision.
8. Saving checkpoints and potentially pushing the trained model to the Hugging Face Hub.
"""

# === Standard Library Imports ===
import io
import os
import json
import logging
import math
import random
from pathlib import Path
from typing import Optional
import argparse
import torch.multiprocessing

# === Third-Party Library Imports ===
# --- Data Handling ---
import pandas as pd
import numpy as np
from datasets import load_dataset
import datasets.utils.logging as datasets_logging
from torch.utils.data import Dataset, DataLoader

# --- Image Processing ---
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFile
import albumentations as alb
from albumentations.pytorch import ToTensorV2

# --- Machine Learning & Deep Learning (Core) ---
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

# --- Hugging Face Ecosystem ---
import transformers
from transformers import CLIPTextModel, CLIPTokenizer # Note: CLIP models seem imported but not directly used in the main logic below (TrOCR is used instead)
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami

# --- Cloud Storage & File IO ---
from minio import Minio
from minio.error import S3Error
from pcache_fileio import fileio # Custom file I/O library, likely for cached access to OSS
# from pcache_fileio.oss_conf import OssConfigFactory # Likely related to pcache_fileio configuration

# --- Optimization & Utilities ---
from functools import lru_cache
from packaging import version
from tqdm.auto import tqdm

# === Alps Specific Imports (Potentially internal framework) ===
# from alps.pytorch.api.utils.web_access import patch_requests # Utility likely for handling network requests within a specific platform/environment
# patch_requests()

# === Configuration & Setup ===
# Ensure minimum diffusers version is met
check_min_version("0.15.0.dev0")

# Set up logging
logger = get_logger(__name__, log_level="INFO")

# Configure PIL to handle large images and truncated files
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Set multiprocessing sharing strategy (important for DataLoader with multiple workers)
torch.multiprocessing.set_sharing_strategy('file_system')

# --- MinIO Configuration (Cloud Object Storage) ---
# Read MinIO connection details from environment variables with defaults
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "play.min.io")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "your-access-key")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "your-secret-key")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "your-bucket")
MINIO_SECURE = os.getenv("MINIO_SECURE", "True").lower() == "true"

# Assuming OSS_PCACHE_ROOT_DIR is defined elsewhere or via environment variable
# Example: OSS_PCACHE_ROOT_DIR = os.getenv("OSS_PCACHE_ROOT_DIR", "oss://your-bucket-name/cache/")
OSS_PCACHE_ROOT_DIR = os.getenv("OSS_PCACHE_ROOT_DIR", "") # Define if needed for download_oss_file_pcache


# === MinIO Client ===
@lru_cache(maxsize=1)
def get_minio_client() -> Minio:
    """
    Creates and returns a MinIO client instance using credentials from environment variables.
    Uses LRU caching to ensure only one client instance is created.

    Returns:
        Minio: An initialized MinIO client object.

    Raises:
        Exception: If the MinIO client fails to initialize.
    """
    try:
        client = Minio(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        logger.info(f"MinIO client created for endpoint: {MINIO_ENDPOINT}, bucket: {MINIO_BUCKET}")
        # Optionally, check if the bucket exists
        # found = client.bucket_exists(MINIO_BUCKET)
        # if not found:
        #     logger.warning(f"MinIO bucket '{MINIO_BUCKET}' not found.")
        return client
    except Exception as e:
        logger.error(f"Failed to create MinIO client: {str(e)}")
        raise


def download_file_minio(file_path: str) -> np.ndarray:
    """
    Downloads a file (expected to be an image) from the configured MinIO bucket
    and decodes it into a NumPy array using OpenCV.

    Args:
        file_path (str): The path (key) of the file within the MinIO bucket.

    Returns:
        np.ndarray: The decoded image as a NumPy array (OpenCV format, BGR).

    Raises:
        S3Error: If a MinIO-specific error occurs during download.
        Exception: If any other error occurs during download or decoding.
    """
    logger.debug(f"Attempting to download from MinIO: {file_path}")
    try:
        client = get_minio_client()

        # Get object data stream from MinIO
        data = client.get_object(MINIO_BUCKET, file_path)

        # Read the data stream into an in-memory bytes buffer
        buffer = io.BytesIO()
        for d in data.stream(32 * 1024): # Read in 32KB chunks
            buffer.write(d)
        buffer.seek(0) # Reset buffer position to the beginning

        # Convert bytes buffer to NumPy array
        file_bytes = np.frombuffer(buffer.read(), dtype=np.uint8)
        # Decode the NumPy array as an image using OpenCV (loads in BGR format)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # flags=1 is equivalent to IMREAD_COLOR

        if img is None:
             raise ValueError(f"cv2.imdecode failed for file: {file_path}. Check if it's a valid image.")
        logger.debug(f"Successfully downloaded and decoded MinIO file: {file_path}")
        return img
    except S3Error as e:
        logger.error(f"MinIO S3 error while downloading {file_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error while processing {file_path} from MinIO: {str(e)}")
        raise


# === Argument Parsing ===
def parse_args():
    """
    Parses command-line arguments for the training script.
    """
    parser = argparse.ArgumentParser(description="Stable Diffusion Inpainting fine-tuning script with text conditioning.")

    # --- Model & Checkpoint Arguments ---
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default=None, required=True,
        help="Path to pretrained Stable Diffusion model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision", type=str, default=None, required=False,
        help="Revision of pretrained model identifier (e.g., a specific commit hash or branch name).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="sd-inpainting-doc-finetuned",
        help="The directory where the trained model checkpoints and logs will be written.",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="The directory where downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help='Whether to resume training from a previous checkpoint. Use "latest" or a specific path.',
    )
    parser.add_argument(
        "--checkpointing_steps", type=int, default=1000,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--checkpoints_total_limit", type=int, default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use Exponential Moving Average (EMA) for the UNet weights."
    )
    parser.add_argument(
        "--non_ema_revision", type=str, default=None, required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Deprecated, use --variant=non_ema with "
            "--pretrained_model_name_or_path instead."
        ),
    )

    # --- Dataset & Data Loading Arguments ---
    parser.add_argument(
        "--dataset_name", type=str, default=None,
        help="The name of the Dataset (from HuggingFace Hub) to train on.",
    )
    parser.add_argument(
        "--dataset_config_name", type=str, default=None,
        help="The config name of the Dataset to use.",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None,
        help="A folder containing the training data (e.g., images and metadata). Used if --dataset_name is not specified.",
    )
    # Note: The script uses a custom dataset `OursDataset` which seems to rely on a hardcoded 'data.csv' and specific
    # download functions (`download_file_minio`, `download_oss_file_pcache`). These arguments might be less relevant
    # unless the dataset loading logic is adapted.
    parser.add_argument(
        "--image_column", type=str, default="image",
        help="The column name in the dataset containing the image paths or data.",
    )
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
        help="Truncate the number of training examples to this value for debugging or faster training.",
    )
    parser.add_argument(
        "--resolution", type=int, default=512,
        help="The resolution for input images. All images will be resized to this size.",
    )
    parser.add_argument(
        "--center_crop", action="store_true", default=False,
        help="Whether to center crop images to the resolution instead of random cropping.",
    )
    parser.add_argument(
        "--random_flip", action="store_true",
        help="Whether to randomly flip images horizontally (often not suitable for text documents).",
    )
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=0,
        help="Number of subprocesses for data loading.",
    )
    # Added based on `OursDataset` usage, might need adjustment
    parser.add_argument(
        "--train_data_csv", type=str, default="data.csv",
        help="Path to the CSV file containing image paths and potentially other metadata for OursDataset.",
    )
    parser.add_argument(
        "--ocr_data_root", type=str, default="ocr_results/",
        help="Root directory or prefix for loading OCR JSON files corresponding to images."
    )

    # --- Training Hyperparameters ---
    parser.add_argument(
        "--train_batch_size", type=int, default=4, # Reduced default due to potentially large inputs
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps", type=int, default=None,
        help="Total number of training steps. Overrides num_train_epochs if set.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate gradients before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, # Adjusted default, often lower for fine-tuning
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--scale_lr", action="store_true", default=False,
        help="Scale the learning rate by num_gpus * grad_accum * batch_size.",
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="constant",
        help='LR scheduler type (e.g., "linear", "cosine", "constant").',
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500,
        help="Number of steps for the warmup in the LR scheduler.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW optimizer beta1.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="AdamW optimizer beta2.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="AdamW optimizer weight decay.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="AdamW optimizer epsilon.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm for clipping.")
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5, # More typical default for Stable Diffusion inference, might not be used directly in this training loop
        help="Guidance scale for classifier-free guidance (more relevant for inference). The training loop provided doesn't explicitly implement CFG loss.")

    # --- Performance & Hardware ---
    parser.add_argument(
        "--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision training (fp16 or bf16).",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true",
        help="Whether to use 8-bit AdamW optimizer from bitsandbytes.",
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true",
        help="Use gradient checkpointing to save memory at the cost of slower backward pass.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true",
        help="Use xformers memory-efficient attention (if available).",
    )
    parser.add_argument(
        "--allow_tf32", action="store_true",
        help="Allow TF32 on Ampere GPUs for potentially faster training.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # --- Logging & Hub ---
    parser.add_argument(
        "--logging_dir", type=str, default="logs",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--report_to", type=str, default="tensorboard",
        help='The integration to report results to (e.g., "tensorboard", "wandb", "comet_ml").',
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the Hugging Face Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, default=None,
        help="The name of the repository on the Hub.",
    )


    args = parser.parse_args()

    # --- Post-processing and Sanity Checks ---
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank # Sync local_rank from environment if set

    # Basic check: Need some form of data input defined
    # if args.dataset_name is None and args.train_data_dir is None and args.train_data_csv is None:
    #     raise ValueError("Need either --dataset_name, --train_data_dir, or --train_data_csv")
    # Adjusted check based on `OursDataset` structure:
    if not os.path.exists(args.train_data_csv):
         raise ValueError(f"Specified --train_data_csv '{args.train_data_csv}' does not exist.")


    # Default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    # Ensure output directory exists (moved here from main for clarity)
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


# === Image/Mask Preprocessing Definitions ===

# Albumentations pipeline for resizing, cropping (implicitly via resize), and normalizing the main image
# Normalization to [-1, 1] is standard for Stable Diffusion VAEs.
image_trans_resize_and_crop = alb.Compose(
    [
        alb.Resize(512, 512), # Resize to target resolution
        alb.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), # Normalize to [-1, 1]
        # ToTensorV2 is applied separately in the dataset __getitem__
    ]
)

# Albumentations pipeline for resizing the mask (no normalization needed for masks)
mask_resize_and_crop = alb.Compose(
    [
        alb.Resize(512, 512), # Resize mask to match image resolution
        # ToTensorV2 is applied separately in the dataset __getitem__
    ]
)

# Albumentations pipeline to convert NumPy arrays to PyTorch tensors (CHW format)
# This is applied after other transformations.
image_trans_to_tensor = alb.Compose(
    [
        ToTensorV2(), # Converts HWC NumPy to CHW Torch Tensor and scales 0-255 to 0-1
    ]
)

# === Text Rendering Helper ===
def draw_text(im_shape, text: str) -> np.ndarray:
    """
    Renders the given text onto a white background image.

    Args:
        im_shape: The shape of the original image (used loosely for font size context, not directly for output size).
        text (str): The text string to render.

    Returns:
        np.ndarray: A NumPy array representing the rendered text image (RGB).
    """
    font_size = 40
    # Ensure a font file is available. Adjust path if necessary or use a system default.
    font_file = 'arialuni.ttf' # Requires 'arialuni.ttf' to be accessible
    if not os.path.exists(font_file):
        # Fallback or error handling for missing font
        logger.warning(f"Font file '{font_file}' not found. Text rendering might fail or use default.")
        # Example: Use a basic PIL font if available
        try:
            font = ImageFont.load_default()
            font_size = 20 # Adjust size for default font
        except IOError:
             logger.error("Default PIL font also not found. Cannot render text.")
             # Return a dummy small black image as fallback
             return np.zeros((20, 50, 3), dtype=np.uint8)
    else:
       font = ImageFont.truetype(font_file, font_size)

    len_text = len(text)
    if len_text == 0:
        # Handle empty text case - create a small blank image
        logger.warning("Attempting to draw empty text. Creating a small blank image.")
        return np.full(((1 + 2) * font_size, 60, 3), 255, dtype=np.uint8) # White image


    # Calculate image size based on text length and font size
    # Use textbbox for potentially better size estimation if using PIL >= 9.0.0
    try:
        # Get bounding box of the text to determine required width/height
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        img_width = text_width + 80 # Add padding
        img_height = text_height + 20 # Add padding
        pos = (40, 10) # Top-left position for drawing text
    except AttributeError:
         # Fallback for older PIL versions
         img_width = (len_text + 2) * font_size
         img_height = 60
         pos = (40, 10) # Fixed position

    # Create a new white background image
    img = Image.new('RGB', (img_width, img_height), color='white')

    # Draw the text onto the image
    draw = ImageDraw.Draw(img)
    draw.text(pos, text, font=font, fill='black') # Black text

    # Convert PIL image to NumPy array (RGB)
    img_np = np.array(img)

    return img_np


# === Location Processing Helper ===
def process_location(location: list, instance_image_size: tuple) -> list:
    """
    Adjusts the bounding box location slightly, potentially expanding it.
    Ensures coordinates stay within image bounds.

    Args:
        location (list): Bounding box coordinates [x_min, y_min, x_max, y_max].
        instance_image_size (tuple): The shape of the image (height, width, channels).

    Returns:
        list: The adjusted bounding box coordinates.
    """
    img_height, img_width = instance_image_size[:2]
    x_min, y_min, x_max, y_max = location

    # Example adjustment: Expand height slightly (by 10% downwards)
    h = y_max - y_min
    y_max_new = min(y_max + h / 10, img_height - 1)

    # Ensure all coordinates are within bounds and valid
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width - 1, x_max)
    y_max_new = min(img_height - 1, y_max_new)

    # Ensure min < max
    if x_max <= x_min: x_max = x_min + 1
    if y_max_new <= y_min: y_max_new = y_min + 1

    return [int(x_min), int(y_min), int(x_max), int(y_max_new)]


# === Mask Generation Helper ===
def generate_mask(im_shape: tuple, ocr_locate: list) -> np.ndarray:
    """
    Generates a binary mask image where the area defined by the OCR location
    bounding box is filled (set to 1 or 255), and the rest is background (0).

    Args:
        im_shape (tuple): The shape of the target mask (width, height). Note: PIL uses (width, height).
        ocr_locate (list): Bounding box [x_min, y_min, x_max, y_max] for the mask region.

    Returns:
        np.ndarray: The generated binary mask as a NumPy array (single channel, 0 or 1).
    """
    # Ensure shape is (width, height) for PIL
    if len(im_shape) == 3: # If (height, width, channels) provided
        width, height = im_shape[1], im_shape[0]
    elif len(im_shape) == 2: # If (height, width) provided
        width, height = im_shape[1], im_shape[0]
    else:
         raise ValueError("Invalid im_shape provided to generate_mask")

    mask = Image.new("L", (width, height), 0) # Create a black background (mode "L" for 8-bit grayscale)
    draw = ImageDraw.Draw(mask)

    # Extract coordinates, ensuring they are integers
    x_min, y_min, x_max, y_max = map(int, ocr_locate)

    # Clamp coordinates to be within image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)

    # Draw a filled rectangle (white) onto the mask
    if x_max > x_min and y_max > y_min: # Ensure valid rectangle dimensions
        draw.rectangle(
            (x_min, y_min, x_max, y_max),
            fill=255, # Use 255 for white (will become 1 after normalization/conversion if needed)
        )
    else:
        logger.warning(f"Invalid rectangle dimensions for mask: {ocr_locate} within shape {(width, height)}")

    # Convert PIL mask to NumPy array. Values will be 0 or 255.
    # The training loop later interpolates this to float [0, 1].
    mask_np = np.array(mask)
    # Optional: Convert to 0/1 explicitly if needed downstream, although interpolation handles 0/255 fine.
    # mask_np = (mask_np / 255.0).astype(np.uint8) # Or float if needed

    return mask_np


# === Masked Image Preparation Helper ===
def prepare_mask_and_masked_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies the mask to the image. The masked area in the output image will be black (0).

    Args:
        image (np.ndarray): The original image (HWC format, e.g., BGR or RGB).
        mask (np.ndarray): The binary mask (HW format, values 0 or 255/1).

    Returns:
        np.ndarray: The masked image, where the masked region is set to zero.
    """
    # Ensure mask is boolean (True where image should be kept, False where it should be masked out)
    # Assuming mask has 0 for background and non-zero (e.g., 255 or 1) for the region to *remove* (inpainting target)
    boolean_mask = mask < 128 # True for background (keep), False for foreground (mask out)

    # Expand mask to have the same number of channels as the image
    boolean_mask_3c = np.stack([boolean_mask] * image.shape[2], axis=-1)

    # Apply the mask: multiply image by the boolean mask (True=1, False=0)
    masked_image = image * boolean_mask_3c

    return masked_image


# === Alternative Data Loading (Using pcache_fileio) ===
def download_oss_file_pcache(my_file: str = "xxx") -> np.ndarray:
    """
    Downloads a file (expected image) using the pcache_fileio library,
    presumably from an OSS (Object Storage Service) source configured elsewhere.

    Args:
        my_file (str): The relative path or identifier for the file within the configured OSS/cache.

    Returns:
        np.ndarray: The decoded image as a NumPy array (OpenCV format, BGR).

    Raises:
        FileNotFoundError: If the file cannot be accessed via pcache_fileio.
        Exception: If decoding fails or other errors occur.
    """
    if not OSS_PCACHE_ROOT_DIR:
        raise ValueError("OSS_PCACHE_ROOT_DIR is not set. Cannot use download_oss_file_pcache.")

    my_file_path = os.path.join(OSS_PCACHE_ROOT_DIR, my_file)
    logger.debug(f"Attempting to download via pcache_fileio: {my_file_path}")
    try:
        # Use fileio context manager to open and read the file
        with fileio.file_io_impl.open(my_file_path, "rb") as fd:
            content = fd.read()

        # Convert raw bytes to NumPy array
        img_buffer = np.frombuffer(content, dtype=np.uint8) # Changed dtype to uint8 for images

        # Decode image using OpenCV
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR) # flags=1 is IMREAD_COLOR

        if img is None:
            raise ValueError(f"cv2.imdecode failed for file loaded via pcache: {my_file}. Check format.")
        logger.debug(f"Successfully downloaded and decoded pcache file: {my_file}")
        return img
    except FileNotFoundError:
        logger.error(f"File not found via pcache_fileio: {my_file_path}")
        raise
    except Exception as e:
        logger.error(f"Error processing file {my_file} using pcache_fileio: {str(e)}")
        raise


# === Custom PyTorch Dataset ===
class OursDataset(Dataset):
    """
    Custom PyTorch Dataset for loading document images, corresponding OCR data,
    generating masks, cropping, and preparing inputs for the inpainting model.

    It expects:
    - A CSV file (`data_csv_path`) listing image paths relative to some base.
    - OCR results stored as JSON files, accessible relative to an `ocr_root` path.
    - Images downloadable via either `download_file_minio` or `download_oss_file_pcache`.
      (The second `__getitem__` uses `download_oss_file_pcache`).
    - A font file for rendering text (`draw_text` function).
    """
    def __init__(self,
                 data_csv_path: str,
                 ocr_root: str,
                 size: int = 512,
                 transform_resize_crop: alb.Compose = None,
                 transform_to_tensor: alb.Compose = None,
                 mask_transform: alb.Compose = None,
                 use_minio: bool = False, # Flag to choose download method
                 crop_scale: int = 256): # Added crop_scale as parameter
        """
        Initializes the dataset.

        Args:
            data_csv_path (str): Path to the CSV file with image paths.
            ocr_root (str): Root path/prefix for finding OCR JSON files.
            size (int): Target resolution for resizing images/masks.
            transform_resize_crop (alb.Compose): Albumentations transform for resizing and normalization.
            transform_to_tensor (alb.Compose): Albumentations transform to convert to tensor.
            mask_transform (alb.Compose): Albumentations transform for resizing masks.
            use_minio (bool): If True, use `download_file_minio`; otherwise, use `download_oss_file_pcache`.
            crop_scale (int): The size of the square crop region around the text.
        """
        self.size = size
        self.data_csv_path = data_csv_path
        self.ocr_root = ocr_root
        self.instance_image_paths = []
        self._load_images_paths() # Load paths from CSV
        self.num_instance_images = len(self.instance_image_paths)
        self._length = self.num_instance_images # Total number of samples
        self.transform_resize_crop = transform_resize_crop
        self.transform_to_tensor = transform_to_tensor
        self.mask_transform = mask_transform
        self.use_minio = use_minio
        self.crop_scale = crop_scale # Store crop scale

        # Basic validation
        if not self.instance_image_paths:
            raise ValueError(f"No image paths loaded from {self.data_csv_path}. Check the file and 'path' column.")
        if not self.transform_resize_crop or not self.transform_to_tensor or not self.mask_transform:
            raise ValueError("Required Albumentations transforms not provided to OursDataset.")

    def _load_images_paths(self):
        """Loads image paths from the specified CSV file."""
        print(f"Loading training file list from: {self.data_csv_path}")
        try:
            # Assuming the CSV has a column named 'path' containing relative image paths
            df = pd.read_csv(self.data_csv_path, low_memory=False)
            if 'path' not in df.columns:
                 raise ValueError(f"CSV file '{self.data_csv_path}' must contain a 'path' column.")
            # Example: Prepend a base directory if paths are relative, adjust as needed
            # base_data_dir = 'data/'
            # self.instance_image_paths = (base_data_dir + df['path']).tolist()
            self.instance_image_paths = df['path'].tolist() # Use paths directly as listed
            print(f"Loaded {len(self.instance_image_paths)} image paths.")
        except FileNotFoundError:
            logger.error(f"Training data CSV file not found: {self.data_csv_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading image paths from {self.data_csv_path}: {str(e)}")
            raise

    # === NOTE: There were two __getitem__ methods defined. ===
    # The first one was simpler and likely overridden or incorrect.
    # It's commented out below. The second, more complex one is assumed to be the intended implementation.

    # def __getitem__(self, index):
    #     """
    #     (Simplified / Potentially Overridden Version)
    #     Loads an instance image, resizes, normalizes, and returns it.
    #     """
    #     example = {}
    #     image_path = self.instance_image_paths[index]
    #     try:
    #         # Choose download method based on the flag
    #         if self.use_minio:
    #             instance_image = download_file_minio(image_path)
    #         else:
    #             # This assumes download_oss_file_pcache is intended here too, adjust if needed
    #             instance_image = download_oss_file_pcache(image_path)

    #         # Basic resize if image is too small (optional, depends on strategy)
    #         h, w, c = instance_image.shape
    #         short_side = min(h, w)
    #         if short_side < 512:
    #             scale_factor = 512 / short_side # Calculate scale factor to make short side 512
    #             new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    #             instance_image = cv2.resize(instance_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    #         # Apply resize, normalization transform
    #         augmented = self.transform_resize_crop(image=instance_image)
    #         instance_image_transformed = augmented["image"]

    #         # Apply tensor conversion transform
    #         augmented_tensor = self.transform_to_tensor(image=instance_image_transformed)
    #         instance_image_tensor = augmented_tensor["image"]

    #         example["instance_images"] = instance_image_tensor # Should match key used in collate_fn

    #         return example
    #     except Exception as e:
    #         logger.error(f"Error processing image at index {index} (path: {image_path}): {str(e)}")
    #         # Return None or raise? Returning None requires careful handling in collate_fn. Raising might be safer.
    #         # Let's try returning a dummy example to avoid crashing the batch, requires collate_fn modification.
    #         # Or simply re-raise:
    #         # raise RuntimeError(f"Failed to process index {index}") from e
    #         # For now, returning a dummy:
    #         logger.warning(f"Returning dummy data for index {index}")
    #         return {"instance_images": torch.zeros(3, self.size, self.size)} # Dummy tensor


    def __getitem__(self, index):
        """
        (Intended / Complex Version)
        Loads an image, its OCR data, selects a text region, generates a mask,
        crops the image/mask/masked_image around the text, renders the text,
        applies transformations, and returns a dictionary of tensors.
        """
        example = {}
        image_path = self.instance_image_paths[index]
        # Construct the expected path for the OCR JSON file
        # Assumes OCR file has same name but .json extension, adjust if structure differs
        base_name = os.path.splitext(image_path)[0]
        ocr_json_path = os.path.join(self.ocr_root, base_name + '.json')

        try:
            # 1. Download the instance image
            if self.use_minio:
                 # Ensure the path format matches MinIO key structure
                 instance_image = download_file_minio(image_path)
            else:
                 # Ensure the path format matches pcache key structure
                 instance_image = download_oss_file_pcache(image_path)

            if instance_image is None: # Check if download failed
                raise ValueError(f"Image download returned None for path: {image_path}")

            # Convert BGR (from cv2) to RGB (for PIL, most transforms)
            instance_image = cv2.cvtColor(instance_image, cv2.COLOR_BGR2RGB)
            original_image_shape = instance_image.shape # (H, W, C)

            # 2. Load corresponding OCR results
            # Using pcache_fileio for OCR JSON as well (adjust if stored differently)
            # Example: Use standard open if OCR files are local
            # if os.path.exists(ocr_json_path):
            #     with open(ocr_json_path, 'r', encoding='utf-8') as f:
            #         content = f.read()
            # else: # Try loading via pcache
            #     logger.warning(f"Local OCR file {ocr_json_path} not found, trying pcache.")
            #     # Assuming pcache needs a path relative to its root, like images
            #     pcache_ocr_path = os.path.join(self.ocr_root, base_name + '.json') # Adjust if pcache needs different prefix
            #     try:
            #         with fileio.file_io_impl.open(pcache_ocr_path, 'r') as f:
            #             content = f.read()
            #     except FileNotFoundError:
            #         raise FileNotFoundError(f"OCR JSON not found locally or via pcache: {ocr_json_path} / {pcache_ocr_path}")
            # Adjusted: Assume OCR path needs prefix removal if using pcache like images
            # This logic depends heavily on how paths and `ocr_root` are structured relative to pcache/minio roots.
            # Let's assume `ocr_json_path` needs to be relative for pcache:
            relative_ocr_path = os.path.join(self.ocr_root, os.path.basename(base_name) + '.json') # Example structure
            try:
                with fileio.file_io_impl.open(relative_ocr_path, 'r') as f:
                    content = f.read()
            except Exception as E:
                 raise FileNotFoundError(f"Could not load OCR json: {relative_ocr_path} Error: {E}")


            ocr_res = json.loads(content)
            # Assuming JSON structure has a 'document' list with 'text', 'score', 'box' keys
            # Adjust parsing based on actual OCR JSON format
            if 'document' not in ocr_res or not isinstance(ocr_res['document'], list):
                 raise ValueError(f"Invalid OCR JSON format in {ocr_json_path}: missing 'document' list.")

            ocr_pd = pd.DataFrame(ocr_res['document'])
            # Basic validation of required columns
            if not all(col in ocr_pd.columns for col in ['text', 'score', 'box']):
                raise ValueError(f"OCR data frame missing required columns ('text', 'score', 'box') in {ocr_json_path}")

            # Filter OCR results by confidence score
            ocr_pd_filtered = ocr_pd[ocr_pd['score'] > 0.8].copy() # Use .copy() to avoid SettingWithCopyWarning

            if ocr_pd_filtered.empty:
                raise ValueError(f"No high-confidence OCR results found in {ocr_json_path} for image {image_path}")

            # 3. Sample one OCR detection
            ocr_pd_sample = ocr_pd_filtered.sample(n=1).iloc[0]
            text = str(ocr_pd_sample['text']) # Ensure text is string
            # 'box' is likely a list of points [[x1, y1], [x2, y2], ...] or similar
            # Convert to [x_min, y_min, x_max, y_max]
            box_coords = ocr_pd_sample['box']
            try:
                x_coords = [p[0] for p in box_coords]
                y_coords = [p[1] for p in box_coords]
                location = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            except (IndexError, TypeError):
                raise ValueError(f"Invalid 'box' format in OCR data: {box_coords}")

            # Process location (optional expansion, clamping)
            location = process_location(location, original_image_shape)
            location_int = np.int32(location) # Ensure integer coords for indexing/drawing

            # 4. Generate mask and masked image (using original image size)
            # Mask shape needs (width, height) for PIL draw
            mask = generate_mask((original_image_shape[1], original_image_shape[0]), location_int)
            masked_image = prepare_mask_and_masked_image(instance_image, mask) # Mask applied to original image

            # 5. Handle potential resizing if image is smaller than crop size
            h, w = original_image_shape[:2]
            short_side = min(h, w)
            if short_side < self.crop_scale:
                scale_factor = self.crop_scale / short_side # Scale up slightly more than needed
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                # Resize image, mask, and masked_image consistently
                instance_image = cv2.resize(instance_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST) # Use NEAREST for masks
                masked_image = cv2.resize(masked_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                # Update location coordinates based on scaling
                location_int = np.int32(np.array(location_int) * scale_factor)
                # Clamp again after scaling
                location_int[0] = max(0, location_int[0])
                location_int[1] = max(0, location_int[1])
                location_int[2] = min(new_w -1, location_int[2])
                location_int[3] = min(new_h -1, location_int[3])


            # 6. Determine cropping coordinates and potentially truncate text
            x1, y1, x2, y2 = location_int
            current_h, current_w = instance_image.shape[:2] # Use potentially resized dimensions

            # Calculate random start points for the crop window, ensuring the text box is included
            # Max possible start X: ensure x_s + crop_scale <= current_w
            max_start_x = max(0, x1 - (self.crop_scale - (x2 - x1))) # Try to center the box if smaller
            max_start_x = min(max_start_x, current_w - self.crop_scale) # Ensure crop fits image width
            min_start_x = max(0, x1) # Crop must start at or after x1 if text box > crop_scale
            min_start_x = min(min_start_x, current_w - self.crop_scale) # Clamp if text box too large near edge

            # Max possible start Y: ensure y_s + crop_scale <= current_h
            max_start_y = max(0, y1 - (self.crop_scale - (y2 - y1))) # Try to center
            max_start_y = min(max_start_y, current_h - self.crop_scale) # Ensure crop fits image height
            min_start_y = max(0, y1) # Crop must start at or after y1
            min_start_y = min(min_start_y, current_h - self.crop_scale) # Clamp

            # Randomly select start coordinates within the valid range
            try:
                # Ensure max >= min before randint
                if max_start_x >= min_start_x:
                     x_s = np.random.randint(min_start_x, max_start_x + 1)
                else:
                     x_s = min_start_x # Fallback if range is invalid
                if max_start_y >= min_start_y:
                    y_s = np.random.randint(min_start_y, max_start_y + 1)
                else:
                     y_s = min_start_y # Fallback

            except ValueError as ive: # Catch potential errors in randint if bounds are wrong
                 logger.warning(f"Issue calculating crop start for index {index}: {ive}. Using fallback starts. Location: {location_int}, Img Size: {(current_h, current_w)}")
                 x_s = max(0, current_w - self.crop_scale) # Crop from right edge
                 y_s = max(0, current_h - self.crop_scale) # Crop from bottom edge


            # Text truncation (Original logic was based on ratios, might be inaccurate)
            # Simpler: If the text box is wider/taller than the crop, we don't need to truncate the *text string* itself.
            # The visual information outside the crop is simply excluded.
            # The rendered text image (`draw_ttf`) should still use the *full* text.
            # No text truncation needed here based on crop.

            # 7. Crop the image, mask, and masked image
            instance_image_crop = instance_image[y_s : y_s + self.crop_scale, x_s : x_s + self.crop_scale, :]
            mask_crop = mask[y_s : y_s + self.crop_scale, x_s : x_s + self.crop_scale]
            masked_image_crop = masked_image[y_s : y_s + self.crop_scale, x_s : x_s + self.crop_scale, :]

            # 8. Render the *full* sampled text onto a separate image
            draw_ttf = draw_text(instance_image_crop.shape, text) # Pass shape for context, use full text

            # 9. Apply transformations (Resize/Norm for image, Resize for mask, ToTensor for all)
            # Note: Cropped images are already `crop_scale`. The 'resize_crop' transform will resize them to `self.size` (e.g., 512).
            # Apply resize/norm to the cropped original image
            augmented = self.transform_resize_crop(image=instance_image_crop)
            instance_image_final = augmented["image"]
            augmented = self.transform_to_tensor(image=instance_image_final)
            instance_image_final = augmented["image"] # CHW tensor, [-1, 1] range

            # Apply resize/norm to the cropped masked image
            augmented = self.transform_resize_crop(image=masked_image_crop)
            masked_image_final = augmented["image"]
            augmented = self.transform_to_tensor(image=masked_image_final)
            masked_image_final = augmented["image"] # CHW tensor, [-1, 1] range

            # Apply resize to the cropped mask
            # Mask should be single channel after transform
            augmented = self.mask_transform(image=mask_crop) # Resize to self.size (e.g., 512x512)
            mask_final = augmented["image"] # HW, values 0 or 255
            # Convert mask to tensor (should become CHW with C=1) and scale to [0, 1]
            mask_final = mask_final.astype(np.float32) / 255.0 # Ensure float for interpolation later
            augmented = self.transform_to_tensor(image=mask_final[:, :, np.newaxis]) # Add channel dim for ToTensorV2
            mask_final = augmented["image"] # CHW tensor (1, H, W), [0, 1] range

            # Convert rendered text image to tensor (assumes text is RGB, normalize if needed)
            # Current draw_text produces RGB, 0-255. TrOCR processor expects normalization.
            # Apply ToTensor first, then normalize if TrOCR processor doesn't handle it.
            augmented = self.transform_to_tensor(image=draw_ttf) # Converts to CHW, scales to [0, 1]
            draw_ttf_tensor = augmented["image"]
            # Example normalization if needed:
            # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # draw_ttf_tensor = normalize(draw_ttf_tensor)
            # However, the main loop uses TrOCRProcessor, which likely handles normalization.

            # 10. Populate example dictionary
            example["instance_images"] = instance_image_final # Target image (resized crop)
            example['mask'] = mask_final                 # Target mask (resized crop, 0/1)
            example['masked_image'] = masked_image_final   # Masked input image (resized crop)
            example['ttf_img'] = draw_ttf_tensor           # Rendered text image tensor

            return example

        except Exception as e:
            logger.error(f"Error processing data at index {index} (path: {image_path}, ocr: {ocr_json_path}): {str(e)}", exc_info=True)
            # Return dummy data to avoid crashing the batch, needs careful handling in collate_fn or main loop.
            logger.warning(f"Returning dummy data for index {index}")
            dummy_img = torch.zeros(3, self.size, self.size)
            dummy_mask = torch.zeros(1, self.size, self.size)
            dummy_ttf = torch.zeros(3, 10, 10) # Placeholder size for TTF
            return {
                "instance_images": dummy_img,
                "mask": dummy_mask,
                "masked_image": dummy_img,
                "ttf_img": dummy_ttf,
                "is_dummy": True # Flag to potentially skip this sample in training
            }


    def __len__(self):
        """Returns the total number of images in the dataset."""
        return self._length

# === Hugging Face Hub Utilities ===
def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None) -> str:
    """
    Constructs the full repository name for the Hugging Face Hub.

    Args:
        model_id (str): The base name of the model repository.
        organization (Optional[str]): The organization name. If None, uses the logged-in user's username.
        token (Optional[str]): Hugging Face Hub authentication token. If None, tries to get it from the environment.

    Returns:
        str: The full repository name (e.g., "username/model_id" or "organization/model_id").
    """
    if token is None:
        token = HfFolder.get_token() # Try to get token from saved credentials
    if organization is None:
        try:
            username = whoami(token)["name"] # Get username associated with the token
            return f"{username}/{model_id}"
        except Exception as e:
            raise ValueError(f"Could not get username from token. Is user logged in (`huggingface-cli login`)? Error: {e}")
    else:
        return f"{organization}/{model_id}"


# === Tensor/NumPy Conversion Utilities ===
def numpy_to_pil(images: np.ndarray) -> list:
    """
    Converts a NumPy image array or a batch of image arrays to a list of PIL Images.
    Assumes input images are in HWC format and pixel values are normalized to [0, 1] or similar range
    that needs to be scaled up to [0, 255].

    Args:
        images (np.ndarray): NumPy array of shape (N, H, W, C) or (H, W, C). Values expected ~[0, 1].

    Returns:
        list: A list of PIL.Image.Image objects.
    """
    if images.ndim == 3:
        images = images[None, ...] # Add batch dimension if single image
    # Denormalize assuming input was [0, 1] -> [0, 255]
    # If input was [-1, 1], denormalize: images = ((images + 1) / 2.0 * 255.0)
    # Based on ToTensorV2, input is likely [0, 1] from dataset/transforms.
    images = (images * 255).round().astype("uint8")

    pil_images = []
    for image in images:
        if image.shape[-1] == 1:
            # Handle grayscale images (single channel)
            pil_images.append(Image.fromarray(image.squeeze(), mode="L"))
        else:
            # Handle RGB images
            pil_images.append(Image.fromarray(image, mode="RGB")) # Assume RGB

    return pil_images


def tensor2im(input_image: torch.Tensor, imtype=np.uint8) -> np.ndarray:
    """
    Converts a PyTorch Tensor representing an image or batch of images into a NumPy array suitable for display/saving.
    Handles denormalization (assumes input tensor is in [-1, 1] range).

    Args:
        input_image (torch.Tensor): Input image tensor (B, C, H, W) or (C, H, W). Expected range [-1, 1].
        imtype (type): The desired data type for the output NumPy array (default: np.uint8).

    Returns:
        np.ndarray: The converted image as a NumPy array (H, W, C), with values in [0, 255].
                    Only returns the first image if input is a batch.
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            # Input is neither Tensor nor ndarray, return as is? Or raise error?
            return input_image # Or raise TypeError

        # Select the first image if it's a batch, move to CPU, convert to float NumPy array
        image_numpy = image_tensor[0].cpu().float().numpy()

        # Convert CHW to HWC format
        image_numpy = np.transpose(image_numpy, (1, 2, 0))

        # Denormalize from [-1, 1] range to [0, 255]
        image_numpy = (image_numpy + 1) / 2.0 * 255.0
    else:
        # If input is already a NumPy array, assume it's correctly formatted HWC, [0, 255]
        # Or should we assume it's [-1, 1] and denormalize? Let's assume it's already displayable.
        image_numpy = input_image

    # Clip values to be safe and convert to the target type
    return np.clip(image_numpy, 0, 255).astype(imtype)


# === Main Training Function ===
def main():
    # --- 1. Argument Parsing and Initial Setup ---
    args = parse_args()

    if args.non_ema_revision is not None:
         # Handle deprecation warning for non_ema_revision
        deprecate(
            "non_ema_revision!=None", "0.15.0",
            message="Downloading 'non_ema' weights via --non_ema_revision is deprecated. Use --variant=non_ema instead."
        )

    # Setup logging directory within the main output directory
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    # Configure Accelerate Project
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit) # Limit total checkpoints saved

    # Initialize Accelerator for distributed training, mixed precision, and logging
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to, # Integrate with TensorBoard/WandB etc.
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    # Setup Python logging -basicConfig ensures logs go to stdout/stderr
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Log Accelerator state (e.g., distributed setup, precision)
    logger.info(accelerator.state, main_process_only=False)

    # Set verbosity for underlying libraries (less noise from datasets/transformers/diffusers)
    if accelerator.is_local_main_process:
        datasets_logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets_logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set seed for reproducibility if provided
    if args.seed is not None:
        set_seed(args.seed)

    # --- 2. Handle Repository Creation (Optional, for Hugging Face Hub) ---
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                # Infer repo name from output directory if not specified
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            # Create the repo on the Hub (or use existing)
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            # Initialize a local git repository synced with the Hub
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            # Add entries to .gitignore to avoid uploading intermediate checkpoints
            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore: gitignore.write("step_*\n")
                if "epoch_*" not in gitignore: gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            # Ensure local output directory exists (already done in parse_args, but safe to repeat)
            os.makedirs(args.output_dir, exist_ok=True)

    # --- 3. Load Models and Scheduler ---
    logger.info("Loading models and scheduler...")
    # Load the noise scheduler from the pretrained model path
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Load TrOCR processor (for preparing text image input) and encoder model (for text feature extraction)
    # Using 'microsoft/trocr-large-printed' - ensure this matches the desired OCR model capability
    logger.info("Loading TrOCR model for text conditioning...")
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed', cache_dir=args.cache_dir)
    trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed', cache_dir=args.cache_dir).encoder

    # Load the AutoencoderKL (VAE) - This path seems specific ('./diffdoc-vae-512/...').
    # Ensure this custom VAE path is correct or replace with VAE from `args.pretrained_model_name_or_path`.
    # Example using VAE from main pretrained model:
    # vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, cache_dir=args.cache_dir)
    logger.info("Loading AutoencoderKL (VAE)...")
    # vae_path = os.path.join("./diffdoc-vae-512/checkpoint-350000/", "vae") # Assuming subfolder structure
    vae_path = args.pretrained_model_name_or_path # Use VAE from the main SD model by default
    vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae", revision=args.revision, cache_dir=args.cache_dir)


    # Load the UNet 2D Condition Model (the core model to be fine-tuned)
    logger.info("Loading UNet model...")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
        revision=args.non_ema_revision, # Use non_ema_revision for initial UNet weights
        cache_dir=args.cache_dir
    )

    # --- 4. Freeze Parameters ---
    # Freeze VAE and TrOCR encoder as they are not being trained
    logger.info("Freezing VAE and TrOCR model parameters.")
    trocr_model.requires_grad_(False)
    vae.requires_grad_(False)
    # UNet parameters remain trainable by default

    # --- 5. Setup EMA (Exponential Moving Average) for UNet (Optional) ---
    if args.use_ema:
        logger.info("Setting up EMA for UNet.")
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet",
            revision=args.revision, # Use the main revision for EMA target
            cache_dir=args.cache_dir
        )
        # Wrap EMA parameters with EMAModel utility
        ema_unet = EMAModel(unet.parameters(), # Track current UNet parameters
                           model_cls=UNet2DConditionModel,
                           model_config=ema_unet.config) # Use EMA model's config


    # --- 6. Enable xFormers Memory Efficient Attention (Optional) ---
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            try:
                logger.info("Enabling xformers memory efficient attention.")
                unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(f"Could not enable xformers memory efficient attention: {e}. Proceeding without it.")
        else:
            logger.warning("xformers is not available. Install it with `pip install xformers` for potential memory savings.")


    # --- 7. Custom Saving Hooks for Accelerator (for diffusers format) ---
    # Required for newer Accelerate versions to save models in the standard diffusers format.
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # Define hooks to be called by accelerator.save_state() and load_state()
        def save_model_hook(models, weights, output_dir):
            """Hook to save models in diffusers format."""
            logger.info(f"Saving models to {output_dir} using custom hook.")
            if args.use_ema:
                 # Save the EMA weights separately
                 ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                 logger.info(f"Saved EMA UNet weights to {os.path.join(output_dir, 'unet_ema')}")

            # Save the main UNet model (the first model in the `models` list)
            if models: # Check if there are models to save
                 models[0].save_pretrained(os.path.join(output_dir, "unet"))
                 logger.info(f"Saved UNet model to {os.path.join(output_dir, 'unet')}")
                 # Pop weights and model to prevent Accelerate from trying to save them again
                 weights.pop()
                 # models.pop() # Don't pop models, Accelerate needs it

        def load_model_hook(models, input_dir):
             """Hook to load models from diffusers format."""
             logger.info(f"Loading models from {input_dir} using custom hook.")
             if args.use_ema:
                 # Load EMA weights
                 logger.info(f"Loading EMA UNet weights from {os.path.join(input_dir, 'unet_ema')}")
                 load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                 ema_unet.load_state_dict(load_model.state_dict())
                 ema_unet.to(accelerator.device) # Ensure EMA model is on correct device
                 del load_model
                 logger.info("EMA UNet weights loaded.")

             # Load the main UNet model
             if models: # Check if there are models to load into
                logger.info(f"Loading UNet model from {os.path.join(input_dir, 'unet')}")
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                # Pop the model placeholder from Accelerator's list and load state into it
                model = models.pop()
                model.register_to_config(**load_model.config) # Update config
                model.load_state_dict(load_model.state_dict()) # Load weights
                del load_model
                logger.info("UNet model loaded.")

        # Register the hooks with Accelerator
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    # --- 8. Enable Gradient Checkpointing (Optional) ---
    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing for UNet.")
        unet.enable_gradient_checkpointing()

    # --- 9. Enable TF32 (Optional, for Ampere GPUs) ---
    if args.allow_tf32:
        logger.info("Allowing TF32 matrix multiplication.")
        torch.backends.cuda.matmul.allow_tf32 = True

    # --- 10. Scale Learning Rate (Optional) ---
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        logger.info(f"Scaled learning rate to: {args.learning_rate}")

    # --- 11. Initialize Optimizer ---
    logger.info("Initializing optimizer...")
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            logger.info("Using 8-bit AdamW optimizer.")
        except ImportError:
            logger.error("bitsandbytes not found. Install with `pip install bitsandbytes`. Falling back to standard AdamW.")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW
        logger.info("Using standard AdamW optimizer.")

    # Create optimizer targeting only the UNet's trainable parameters
    optimizer = optimizer_cls(
        unet.parameters(), # Only pass UNet parameters
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # --- 12. Define Custom Collate Function ---
    def collate_fn_ours(examples):
        """
        Custom collate function to handle batches from OursDataset.
        Filters out dummy examples if they exist.
        Stacks tensors for images, masks, masked images, and text images.
        """
        # Filter out potential dummy examples marked by the dataset
        valid_examples = [ex for ex in examples if not ex.get("is_dummy", False)]

        if not valid_examples:
            # Handle case where the entire batch consists of dummy examples
            logger.warning("Entire batch consists of dummy examples. Skipping batch.")
            # Returning None or an empty dict signals to the training loop to skip this batch.
            return None

        # Stack pixel values (target images)
        pixel_values = torch.stack([example["instance_images"] for example in valid_examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        # Stack masks
        masks = torch.stack([example["mask"] for example in valid_examples])
        masks = masks.to(memory_format=torch.contiguous_format).float() # Ensure masks are float

        # Stack masked images (input condition)
        masked_images = torch.stack([example["masked_image"] for example in valid_examples])
        masked_images = masked_images.to(memory_format=torch.contiguous_format).float()

        # Collect text image tensors (TTF renders) - might have variable size if not resized/padded
        # The TrOCR processor should handle padding/resizing. Collect as a list.
        ttf_imgs = [example["ttf_img"] for example in valid_examples]
        # Note: TrOCR processor applied later in the training loop handles batching/padding of TTF images

        # Return batch dictionary
        batch = {
            "pixel_values": pixel_values,   # Target original images (cropped, resized)
            "masks": masks,                 # Target masks (cropped, resized)
            "masked_images": masked_images, # Input masked images (cropped, resized)
            "ttf_images": ttf_imgs          # List of TTF image tensors (variable size)
        }
        return batch

    # --- 13. Create Dataset and DataLoader ---
    logger.info("Creating dataset and dataloader...")
    # Instantiate the custom dataset
    train_dataset = OursDataset(
        data_csv_path=args.train_data_csv,
        ocr_root=args.ocr_data_root,
        size=args.resolution, # Use resolution argument
        transform_resize_crop=image_trans_resize_and_crop, # Pass defined transforms
        transform_to_tensor=image_trans_to_tensor,
        mask_transform=mask_resize_and_crop,
        use_minio=False, # Choose data source (False for pcache, True for MinIO - adjust as needed)
        crop_scale=256   # Set crop size (can be made configurable via args)
    )

    # Create the DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True, # Shuffle data each epoch
        collate_fn=collate_fn_ours, # Use custom collate function
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True # Might speed up data transfer to GPU
    )

    # --- 14. Setup Learning Rate Scheduler ---
    logger.info("Setting up learning rate scheduler...")
    # Calculate total training steps if not provided
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        logger.info(f"Calculated max_train_steps: {args.max_train_steps} ({args.num_train_epochs} epochs)")

    # Get the scheduler function from diffusers library
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # --- 15. Prepare Components with Accelerator ---
    logger.info("Preparing models, optimizer, dataloader, and scheduler with Accelerator...")
    # Accelerator handles moving models/data to the correct device(s) and wrapping for DDP/FSDP
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Move EMA model to the correct device if used
    if args.use_ema:
        ema_unet.to(accelerator.device)

    # --- 16. Set Weight Data Type for Inference Models ---
    # Move VAE and TrOCR (used only for inference) to the correct device and potentially cast to lower precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        logger.info("Using float16 for VAE and TrOCR.")
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        logger.info("Using bfloat16 for VAE and TrOCR.")

    logger.info("Moving VAE and TrOCR to device and setting dtype...")
    vae.to(accelerator.device, dtype=weight_dtype)
    trocr_model.to(accelerator.device, dtype=weight_dtype)

    # Get VAE downscaling factor (used for resizing masks to latent space dimensions)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    logger.info(f"VAE scale factor: {vae_scale_factor}")


    # --- 17. Recalculate Training Steps/Epochs (Post-Accelerator Prepare) ---
    # The dataloader size might change with distributed training, so recalculate steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps: # If we calculated max_train_steps based on initial dataloader length
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        logger.info(f"Recalculated max_train_steps after Accelerator: {args.max_train_steps}")
    # Recalculate number of epochs based on potentially updated max_train_steps
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    logger.info(f"Effective number of epochs: {args.num_train_epochs}")


    # --- 18. Initialize Trackers (TensorBoard/WandB) ---
    if accelerator.is_main_process:
        # Initialize trackers defined in `log_with` argument (e.g., "tensorboard")
        # Pass hyperparameter configuration for logging
        accelerator.init_trackers("stable_diffusion_inpainting_doc", config=vars(args))
        logger.info("Initialized trackers.")


    # --- 19. Training Loop ---
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Starting Training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, dist & accum) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # --- 20. Resume from Checkpoint (Optional) ---
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint) # Use specific checkpoint folder name
        else:
            # Find the most recent checkpoint directory
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1])) # Sort by step number
            path = dirs[-1] if len(dirs) > 0 else None

        if path and os.path.isdir(os.path.join(args.output_dir, path)):
            resume_path = os.path.join(args.output_dir, path)
            logger.info(f"Resuming from checkpoint {resume_path}")
            accelerator.load_state(resume_path) # Load state using Accelerator (handles custom hooks)
            # Extract global step from checkpoint path name
            global_step = int(path.split("-")[1])
            # Calculate how many steps/epochs have already been completed
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
            logger.info(f"Resumed training from step {global_step}, epoch {first_epoch}.")
        else:
            logger.warning(f"Checkpoint '{args.resume_from_checkpoint}' not found or invalid. Starting a new training run.")
            args.resume_from_checkpoint = None # Reset flag if checkpoint not found

    # Initialize progress bar
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                       disable=not accelerator.is_local_main_process, # Show only on main process
                       desc="Training Steps")

    # --- Main Epoch and Step Loop ---
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train() # Set UNet to training mode
        train_loss = 0.0 # Accumulate loss for logging

        for step, batch in enumerate(train_dataloader):

            # Handle skipping steps if resuming from checkpoint
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1) # Update progress bar even when skipping
                continue

            # Handle cases where collate_fn returns None (e.g., all dummy examples)
            if batch is None:
                logger.warning(f"Skipping step {step} due to empty batch.")
                continue

            # --- Forward Pass ---
            # Use accelerator.accumulate context manager for gradient accumulation
            with accelerator.accumulate(unet):
                # --- a. Prepare Text Conditioning ---
                # Process the list of TTF image tensors using the TrOCR processor
                # This handles padding, resizing, normalization needed for the TrOCR model
                try:
                     # Ensure ttf_images are PIL Images or tensors processor can handle
                     # If they are tensors [0,1], may need conversion or processor adjustment
                     # Assuming processor expects PIL Images or raw tensors:
                     processed_text_input = processor(images=batch["ttf_images"], return_tensors="pt").pixel_values
                except Exception as proc_e:
                     logger.error(f"Error during TrOCR processing at step {step+1}: {proc_e}")
                     # Skip batch or handle error appropriately
                     continue

                processed_text_input = processed_text_input.to(accelerator.device, dtype=weight_dtype)

                # Get text embeddings from the TrOCR encoder (frozen)
                # No gradient calculation needed for this part
                with torch.no_grad():
                    ocr_feature = trocr_model(processed_text_input)
                    ocr_embeddings = ocr_feature.last_hidden_state # Use the last hidden state as conditioning

                # --- b. Prepare Image Latents and Mask ---
                # Convert target images, masked images to latent space using VAE (frozen)
                # No gradient calculation needed for VAE encoding
                with torch.no_grad():
                    # Encode original images -> target latents
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor # Apply scaling factor

                    # Encode masked images -> condition latents
                    masked_image_latents = vae.encode(batch["masked_images"].to(dtype=weight_dtype)).latent_dist.sample()
                    masked_image_latents = masked_image_latents * vae.config.scaling_factor

                # Prepare the mask for the latent space
                # Resize the image-space mask (0/1, size HxW) down to latent-space size (hxw)
                mask = F.interpolate(
                    batch["masks"].to(dtype=weight_dtype), # Ensure mask is on device and correct dtype
                    size=(latents.shape[2], latents.shape[3]) # Target latent dimensions (H/sf, W/sf)
                )
                # Mask should be [0, 1] float tensor on the correct device

                # --- c. Prepare Noisy Latents (Forward Diffusion) ---
                # Sample random noise matching the latent dimensions
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample random timesteps for each image in the batch
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise scheduler and timesteps
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # --- d. UNet Prediction ---
                # Prepare the input for the UNet: concatenate noisy latents, mask, and masked image latents
                # Input channels: latent_channels + mask_channels(1) + masked_latent_channels
                model_input_latents = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)

                # Get the text embedding conditioning (already detached and on correct device/dtype)
                encoder_hidden_states = ocr_embeddings.detach()

                # Predict the noise residual (or v_prediction, depending on scheduler) using the UNet
                # The UNet takes the concatenated latents, timestep, and text embeddings as input
                model_pred = unet(model_input_latents, timesteps, encoder_hidden_states).sample

                # --- e. Calculate Loss ---
                # Determine the target for the loss based on the scheduler's prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise # Predict the noise added
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps) # Predict velocity
                else:
                    raise ValueError(f"Unsupported prediction type: {noise_scheduler.config.prediction_type}")

                # Calculate the Mean Squared Error loss between the model's prediction and the target
                # Ensure both prediction and target are float32 for loss calculation consistency
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # --- f. Backpropagation and Optimization ---
                # Accumulate loss for logging (average across devices and gradient accumulation steps)
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Perform backward pass (calculates gradients). Accelerator handles scaling for mixed precision.
                accelerator.backward(loss)

                # Clip gradients if enabled and if it's time for an optimizer step
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)

                # Perform optimizer step (updates UNet weights)
                optimizer.step()
                # Update learning rate based on the scheduler
                lr_scheduler.step()
                # Zero out gradients for the next accumulation cycle
                optimizer.zero_grad()

            # --- Post-Step Operations ---
            # Check if an optimization step was performed (gradients were synced and optimizer stepped)
            if accelerator.sync_gradients:
                # Update EMA model parameters if EMA is enabled
                if args.use_ema:
                    ema_unet.step(unet.parameters()) # Pass current UNet parameters to EMA

                progress_bar.update(1) # Increment progress bar
                global_step += 1 # Increment global step counter

                # Log training loss
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0 # Reset accumulated loss for next logging interval

                # --- Checkpointing ---
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process: # Save only on the main process
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # Use accelerator.save_state to save model, optimizer, scheduler, etc.
                        # Custom hooks handle saving UNet/EMA in diffusers format.
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint state to {save_path}")

                        # Optional: Save a pipeline version for easy inference testing
                        # This requires loading the weights into a pipeline object
                        # unet_to_save = accelerator.unwrap_model(unet)
                        # if args.use_ema:
                        #      ema_unet.copy_to(unet_to_save.parameters()) # Copy EMA weights for saving
                        # pipeline = StableDiffusionPipeline.from_pretrained(
                        #      args.pretrained_model_name_or_path,
                        #      unet=unet_to_save,
                        #      vae=vae, # Use the loaded VAE
                        #      revision=args.revision,
                        #      torch_dtype=weight_dtype, # Use appropriate dtype
                        #      # text_encoder and tokenizer are not strictly needed if TrOCR is the condition
                        # )
                        # # Pipeline saving needs text_encoder/tokenizer, might need dummy ones or modification
                        # # pipeline.save_pretrained(save_path)
                        # logger.info(f"Saved pipeline checkpoint (potentially requires text_encoder/tokenizer setup) to {save_path}")


            # Log step loss and learning rate to the progress bar
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Check if max training steps have been reached
            if global_step >= args.max_train_steps:
                logger.info("Reached max_train_steps. Exiting training loop.")
                break # Exit inner loop (steps)

        # Check again after epoch end (in case max_steps aligns with epoch end)
        if global_step >= args.max_train_steps:
            break # Exit outer loop (epochs)


    # --- 21. End of Training ---
    logger.info("Training finished.")
    # Ensure all processes wait until completion
    accelerator.wait_for_everyone()

    # --- 22. Save Final Model ---
    if accelerator.is_main_process:
        # Unwrap the model potentially wrapped by Accelerator (DDP, FSDP)
        unet_final = accelerator.unwrap_model(unet)
        if args.use_ema:
             # Copy EMA weights to the UNet model before saving the final version
             logger.info("Copying EMA weights to final UNet model.")
             ema_unet.copy_to(unet_final.parameters())

        # Save the final UNet model in diffusers format
        unet_final.save_pretrained(os.path.join(args.output_dir, "unet"))
        logger.info(f"Saved final UNet model to {os.path.join(args.output_dir, 'unet')}")

        # Optional: Save the full pipeline
        # Again, requires handling the text_encoder/tokenizer part if saving a standard SD pipeline
        # pipeline_final = StableDiffusionPipeline.from_pretrained(
        #     args.pretrained_model_name_or_path,
        #     unet=unet_final,
        #     vae=vae,
        #     revision=args.revision,
        #     torch_dtype=weight_dtype,
        #     # Provide text_encoder and tokenizer if saving standard pipeline
        # )
        # pipeline_final.save_pretrained(args.output_dir)
        # logger.info(f"Saved final pipeline to {args.output_dir}")


        # --- 23. Push to Hub (Optional) ---
        if args.push_to_hub:
            logger.info("Pushing final model to Hugging Face Hub...")
            # Use the repository object initialized earlier
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)
            logger.info(f"Model pushed to Hub repository: {repo_name}")

    # Clean up accelerator resources
    accelerator.end_training()
    logger.info("Accelerator training ended.")


# === Script Entry Point ===
if __name__ == "__main__":
    main()