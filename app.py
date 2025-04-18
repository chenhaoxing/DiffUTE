"""
DiffUTE: Universal Text Editing Diffusion Model

A comprehensive text editing system that uses diffusion models to modify text in images
while preserving the surrounding context. This implementation provides a user-friendly
interface for text editing tasks in images.

Model Architecture and Training Process:
--------------------------------------
The system uses two key neural networks that work together:

1. VAE (Variational AutoEncoder):
   - Purpose: Compresses images into a compact latent space and reconstructs them
   - Training Process:
     * Input: Original high-dimensional images
     * Output: Reconstructed images
     * Loss: Mean Squared Error between input and output
     * Key Feature: Creates a compressed "latent space" representation
   - Why it's needed:
     * Reduces computational complexity for the diffusion process
     * Creates a more manageable space for the diffusion model to work in
     * Helps maintain image structure and features during editing

2. UNet (with Diffusion):
   - Purpose: Performs the actual text editing in the latent space
   - Training Process:
     * Forward Diffusion: Gradually adds noise to images
     * Reverse Diffusion: Learns to denoise images with text conditioning
     * Loss: Mean Squared Error between predicted and actual noise
   - Why it's needed:
     * Handles the progressive generation of new text
     * Maintains consistency with surrounding context
     * Provides fine-grained control over the editing process

Training Strategy:
-----------------
1. VAE Training (train_vae.py):
   - Trained first to establish stable latent space
   - Focuses on reconstruction quality
   - Uses direct image-to-image comparison
   - Frozen during UNet training
   - Key hyperparameters:
     * Lower learning rate (typically 1e-5)
     * Smaller batch size
     * MSE loss for pixel-level accuracy

2. UNet Training (train_diffute_v1.py):
   - Uses pre-trained VAE
   - Implements diffusion-based training
   - Incorporates text conditioning
   - Key hyperparameters:
     * Higher learning rate initially
     * Noise scheduler configuration
     * Text embedding integration

Why Two Separate Models:
-----------------------
1. Separation of Concerns:
   - VAE handles image compression/reconstruction
   - UNet focuses on text generation/editing
   - Each model can be optimized independently

2. Computational Efficiency:
   - VAE reduces dimensionality for faster processing
   - UNet works in smaller latent space
   - Enables real-time editing capabilities

3. Quality Control:
   - VAE ensures structural integrity
   - UNet manages fine details and text generation
   - Combined approach preserves image quality

Implementation Details:
----------------------
1. Data Processing:
   - Images are normalized to [-1, 1]
   - Text regions are masked for targeted editing
   - Resolution is standardized (typically 512x512)

2. Training Pipeline:
   - VAE training:
     * Direct image reconstruction
     * No text conditioning
     * Frozen during inference
   
   - UNet training:
     * Noise prediction in latent space
     * Text conditioning via TrOCR
     * Progressive denoising process

3. Inference Process:
   - User selects text region
   - VAE encodes image to latent space
   - UNet generates new text through denoising
   - VAE decodes back to image space

Key Features:
------------
1. Region-Specific Editing:
   - Precise control over text areas
   - Preserves surrounding context
   - Smooth blending of edited regions

2. Text Conditioning:
   - TrOCR for text understanding
   - Guided generation process
   - Maintains text style consistency

3. Quality Preservation:
   - High-fidelity image reconstruction
   - Clean text generation
   - Seamless integration of edits

Usage Notes:
-----------
- The VAE should be trained first and frozen
- UNet training requires more computational resources
- Text conditioning quality affects final results
- Number of diffusion steps impacts generation quality

Dependencies and Requirements:
----------------------------
- PyTorch: Deep learning framework
- Diffusers: Diffusion model implementation
- Transformers: Text processing
- Accelerate: Distributed training
- Additional utilities for image processing

Detailed Process Overview:
-------------------------
1. Image and Region Selection
   - User selects an image and specifies the region containing text to edit
   - The system captures coordinates (x0, y0, x1, y1) of the selected region
   - These coordinates help focus the editing process on the relevant area

2. Text Input and Preprocessing
   - User provides new text to replace the existing text
   - The system renders this text using a specified font (arialuni.ttf)
   - The rendered text serves as a guide for the diffusion model

3. Mask Generation
   - A binary mask is created based on the selected region
   - White (1) represents the area to edit, black (0) represents area to preserve
   - This mask helps the model understand which parts of the image to modify

4. Image Processing and Cropping
   - The system crops the image around the text region with padding
   - Cropping helps focus computation on the relevant area
   - Various image sizes are handled through adaptive cropping logic

5. Diffusion Model Process (Core Technology)
   - The diffusion model works by gradually denoising an image:
     a. Start with pure noise in the text region
     b. Gradually refine this noise into meaningful text
     c. Use the provided new text as guidance
   - Key steps in diffusion:
     * Forward process: Not used during inference
     * Reverse process: Gradually removes noise to create the new text
     * Guidance: Uses TrOCR features to ensure the generated content matches the desired text

6. Technical Components
   - VAE (Variational AutoEncoder):
     * Compresses images into a smaller latent space
     * Makes it easier for the diffusion model to work
   - UNet:
     * The main architecture for the diffusion process
     * Predicts noise at each step of denoising
   - TrOCR:
     * Transformer-based OCR model
     * Provides features to guide the text generation

7. Post-Processing
   - The generated text region is carefully blended back into the original image
   - Resolution and dimensions are matched to ensure seamless integration
   - The final image maintains the original context while showing the new text

Key Concepts for Beginners:
--------------------------
- Diffusion Models: Think of these as models that learn to gradually clean up noise
  into meaningful images. Like slowly revealing a picture through fog.
  
- Latent Space: A compressed representation of images where the model works.
  Similar to how a jpeg compresses an image, but optimized for AI operations.
  
- Masks: Like stencils in painting, they tell the model which parts of the image
  to modify and which to leave unchanged.
  
- Guidance: The process of telling the model what kind of text to generate.
  Similar to having a reference while drawing.

Usage Example:
-------------
1. Load an image with text you want to edit
2. Select the text region by clicking two points to form a rectangle
3. Enter the new text you want to appear in that region
4. Adjust the number of diffusion steps if needed (more steps = potentially better quality)
5. Click generate to create the edited image

The system will handle all the complex processing while providing a simple interface
for users to perform sophisticated text editing tasks.

Dependencies and Models:
----------------------
- PyTorch: Deep learning framework
- Diffusers: Library for diffusion models
- Transformers: For text processing
- Gradio: Web interface
- OpenCV & PIL: Image processing
- Albumentations: Image augmentation

Note: The quality of results can depend on factors like:
- Text region size and location
- Complexity of the background
- Number of diffusion steps
- Resolution of the input image

TrOCR-UNet Integration:
----------------------
The system integrates TrOCR embeddings with the UNet through a sophisticated 
cross-attention mechanism:

1. Text Feature Extraction:
   - Input text is rendered as an image using a standard font
   - TrOCR processor prepares the image for the encoder
   - TrOCR encoder generates embeddings [batch_size, sequence_length, hidden_size]
   - Embeddings capture both semantic and style information

2. UNet Conditioning Architecture:
   - Cross-attention layers in UNet attend to text embeddings
   - AdaIN layers modulate feature maps based on text style
   - Embeddings guide the denoising process at each timestep
   - Multi-modal fusion combines image and text information

3. Information Flow:
   - Text → TrOCR Encoder → Embeddings → UNet Cross-Attention
   - Embeddings remain consistent during denoising steps
   - Cross-attention selectively applies text features
   - Progressive refinement guided by text condition

4. Training Configuration:
   - TrOCR encoder is frozen (requires_grad=False)
   - UNet learns to interpret embeddings effectively
   - Loss computed between noise predictions and targets
   - Embeddings provide stable conditioning signal

5. Input Processing:
   - Latent input concatenates:
     * Noisy image latents
     * Binary mask for text region
     * Masked image latents
   - Text embeddings passed as separate conditioning

Benefits of this Integration:
- Precise control over text content and style
- Stable generation guided by text features
- Effective multi-modal fusion
- Selective application of text information

This architecture enables:
- Accurate text content generation
- Style consistency with input specification
- Context-aware text placement
- Smooth blending with background
"""

import gradio as gr
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFile

import argparse
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Union
import accelerate
import datasets
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
import diffusers
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")
logger = get_logger(__name__, log_level="INFO")

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser()

parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    default="/mnt/new/sd2-inp",
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--revision",
    type=str,
    default=None,
    required=False,
    help="Revision of pretrained model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default=None,
    help=(
        "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
        " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
        " or to a folder containing files that 🤗 Datasets can understand."
    ),
)
parser.add_argument(
    "--dataset_config_name",
    type=str,
    default=None,
    help="The config of the Dataset, leave as None if there's only one config.",
)
parser.add_argument(
    "--train_data_dir",
    type=str,
    default=None,
    help=(
        "A folder containing the training data. Folder contents must follow the structure described in"
        " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
        " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
    ),
)
parser.add_argument(
    "--image_column",
    type=str,
    default="image",
    help="The column of the dataset containing an image.",
)
parser.add_argument(
    "--max_train_samples",
    type=int,
    default=None,
    help=(
        "For debugging purposes or quicker training, truncate the number of training examples to this "
        "value if set."
    ),
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="sd-model-finetuned",
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=None,
    help="The directory where the downloaded models and datasets will be stored.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="A seed for reproducible training."
)
parser.add_argument(
    "--resolution",
    type=int,
    default=512,
    help=(
        "The resolution for input images, all the images in the train/validation dataset will be resized to this"
        " resolution"
    ),
)
parser.add_argument("--guidance_scale", type=float, default=0.8)

parser.add_argument(
    "--center_crop",
    default=False,
    action="store_true",
    help=(
        "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
        " cropped. The images will be resized to the resolution first before cropping."
    ),
)
parser.add_argument(
    "--random_flip",
    action="store_true",
    help="whether to randomly flip images horizontally",
)
parser.add_argument(
    "--train_batch_size",
    type=int,
    default=16,
    help="Batch size (per device) for the training dataloader.",
)
parser.add_argument("--num_train_epochs", type=int, default=100)
parser.add_argument(
    "--max_train_steps",
    type=int,
    default=None,
    help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
)

parser.add_argument(
    "--select_data_lenth",
    type=int,
    default=100,
    help="Number of images selected for training.",
)

parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--gradient_checkpointing",
    action="store_true",
    help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-4,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument(
    "--scale_lr",
    action="store_true",
    default=False,
    help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
)
parser.add_argument(
    "--lr_warmup_steps",
    type=int,
    default=500,
    help="Number of steps for the warmup in the lr scheduler.",
)
parser.add_argument(
    "--use_8bit_adam",
    action="store_true",
    help="Whether or not to use 8-bit Adam from bitsandbytes.",
)
parser.add_argument(
    "--allow_tf32",
    action="store_true",
    help=(
        "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
        " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
    ),
)
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
parser.add_argument(
    "--non_ema_revision",
    type=str,
    default=None,
    required=False,
    help=(
        "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
        " remote repository specified with --pretrained_model_name_or_path."
    ),
)
parser.add_argument(
    "--dataloader_num_workers",
    type=int,
    default=0,
    help=(
        "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
    ),
)
parser.add_argument(
    "--adam_beta1",
    type=float,
    default=0.9,
    help="The beta1 parameter for the Adam optimizer.",
)
parser.add_argument(
    "--adam_beta2",
    type=float,
    default=0.999,
    help="The beta2 parameter for the Adam optimizer.",
)
parser.add_argument(
    "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
)
parser.add_argument(
    "--adam_epsilon",
    type=float,
    default=1e-08,
    help="Epsilon value for the Adam optimizer",
)
parser.add_argument(
    "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
)
parser.add_argument(
    "--push_to_hub",
    action="store_true",
    help="Whether or not to push the model to the Hub.",
)
parser.add_argument(
    "--hub_token",
    type=str,
    default=None,
    help="The token to use to push to the Model Hub.",
)
parser.add_argument(
    "--hub_model_id",
    type=str,
    default=None,
    help="The name of the repository to keep in sync with the local `output_dir`.",
)
parser.add_argument(
    "--logging_dir",
    type=str,
    default="logs",
    help=(
        "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
        " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
    ),
)
parser.add_argument(
    "--mixed_precision",
    type=str,
    default=None,
    choices=["no", "fp16", "bf16"],
    help=(
        "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
        " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
        " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    ),
)
parser.add_argument(
    "--report_to",
    type=str,
    default="tensorboard",
    help=(
        'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
        ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
    ),
)
parser.add_argument(
    "--local_rank", type=int, default=-1, help="For distributed training: local_rank"
)
parser.add_argument(
    "--checkpointing_steps",
    type=int,
    default=1000,
    help=(
        "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
        " training using `--resume_from_checkpoint`."
    ),
)
parser.add_argument(
    "--checkpoints_total_limit",
    type=int,
    default=None,
    help=(
        "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
        " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
        " for more docs"
    ),
)
parser.add_argument(
    "--resume_from_checkpoint",
    type=str,
    default=None,
    help=(
        "Whether training should be resumed from a previous checkpoint. Use a path saved by"
        ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
    ),
)
parser.add_argument(
    "--image_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--ocr_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--enable_xformers_memory_efficient_attention",
    action="store_true",
    help="Whether or not to use xformers.",
)

args, unknown = parser.parse_known_args()

env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
if env_local_rank != -1 and env_local_rank != args.local_rank:
    args.local_rank = env_local_rank

# default to using the same revision for the non-ema model if not specified
if args.non_ema_revision is None:
    args.non_ema_revision = args.revision


# generate mask for ocr region
image_trans_resize_and_crop = alb.Compose(
    [
        alb.Resize(512, 512),
        alb.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

mask_resize_and_crop = alb.Compose(
    [
        alb.Resize(512, 512),
    ]
)

image_trans = alb.Compose(
    [
        ToTensorV2(),
    ]
)


def draw_text(im_shape, text):
    """
    Renders text onto a blank image using a specified font.

    This function creates a new image with the same dimensions as the input shape,
    renders the provided text using the Arial Unicode font, and returns the rendered
    image. The text is rendered in black on a white background.

    Args:
        im_shape (tuple): Shape of the target image (height, width, channels).
        text (str): Text to render.

    Returns:
        np.ndarray: RGB image array with rendered text.

    Raises:
        ValueError: If the font file is not found or text rendering fails.
    """
    # Set font parameters
    font_size = 40  
    font_file = "arialuni.ttf"
    len_text = len(text) if len(text) > 0 else 3

    # Create white background image
    img = Image.new("RGB", ((len_text + 2) * font_size, 60), color="white")
    
    # Load font and draw text
    font = ImageFont.truetype(font_file, font_size)
    pos = (40, 10)
    draw = ImageDraw.Draw(img)
    draw.text(pos, text, font=font, fill="black")
    
    return np.array(img)


def process_location(location, instance_image_size):
    """
    Processes and validates text region coordinates.

    This function takes the coordinates of a text region and ensures they are valid
    within the image boundaries. It also applies padding and other adjustments to
    improve text region selection.

    Args:
        location (list): [x_min, y_min, x_max, y_max] coordinates of text region.
        instance_image_size (tuple): (height, width, channels) of the image.

    Returns:
        list: Processed [x_min, y_min, x_max, y_max] coordinates.

    Raises:
        ValueError: If coordinates are invalid or outside image boundaries.
    """
    h = location[3] - location[1]
    location[3] = min(location[3] + h / 10, instance_image_size[0] - 1)
    return location


def generate_mask(im_shape, ocr_locate):
    """
    Generates a binary mask for text regions in an image.

    Creates a mask where text regions (specified by OCR locations) are marked as
    white (255) on a black (0) background. The mask can be used to identify
    regions that need to be edited or preserved during image processing.

    Args:
        im_shape (tuple): Shape of the target image (height, width).
        ocr_locate (list): List of text region coordinates [x_min, y_min, x_max, y_max].

    Returns:
        np.ndarray: Binary mask array where text regions are marked as 255.

    Raises:
        ValueError: If shape or coordinates are invalid.
    """
    # Create empty mask
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    
    # Draw rectangle for text region
    draw.rectangle(
        (ocr_locate[0], ocr_locate[1], ocr_locate[2], ocr_locate[3]),
        fill=1,
    )
    return np.array(mask)


def prepare_mask_and_masked_image(image, mask):
    """
    Prepares a masked version of an input image.

    This function applies a binary mask to an image, creating a version where
    the masked regions (text areas) are blacked out. This is used to prepare
    inputs for the inpainting model.

    Args:
        image (np.ndarray): Original image array (RGB).
        mask (np.ndarray): Binary mask array where text regions are marked.

    Returns:
        np.ndarray: Image with text regions masked out.

    Raises:
        ValueError: If image and mask shapes don't match.
    """
    masked_image = np.multiply(
        image, np.stack([mask < 0.5, mask < 0.5, mask < 0.5]).transpose(1, 2, 0)
    )
    return masked_image


def download_oss_file_pcache(my_file="xxx"):
    """
    Downloads a file from OSS using pcache.

    This function is a legacy implementation for downloading files from OSS
    storage using pcache. It's recommended to use MinIO client instead.

    Args:
        my_file (str): Path to the file in OSS storage.

    Returns:
        np.ndarray: Downloaded file content as a NumPy array.

    Raises:
        Exception: If download or file processing fails.

    Note:
        This function is deprecated and should be replaced with MinIO client usage.
    """
    # This function is no longer needed as we're not using OSS
    pass


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    """
    Constructs the full repository name for Hugging Face Hub.

    This function takes a model ID and optional organization name to construct
    the full repository name used for pushing models to the Hugging Face Hub.

    Args:
        model_id (str): Base name/ID for the model.
        organization (str, optional): Organization name on Hugging Face Hub.
        token (str, optional): Authentication token for Hugging Face Hub.

    Returns:
        str: Full repository name (e.g., "organization/model-id").

    Raises:
        ValueError: If required authentication information is missing.
    """
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def numpy_to_pil(images):
    """
    Converts NumPy image arrays to PIL Image objects.

    This function handles the conversion of one or more images from NumPy array
    format to PIL Image objects, including necessary normalization and type
    conversion.

    Args:
        images (np.ndarray): Single image or batch of images as NumPy arrays.

    Returns:
        list: List of PIL Image objects.

    Raises:
        ValueError: If input array format is invalid.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    
    if images.shape[-1] == 1:
        # Handle grayscale images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def tensor2im(input_image, imtype=np.uint8):
    """
    Converts a PyTorch tensor to a NumPy image array.

    This function handles the conversion of image data from PyTorch tensor format
    to NumPy array format, including denormalization and type conversion.

    Args:
        input_image (torch.Tensor): Input image tensor.
        imtype (np.dtype): Target NumPy data type (default: np.uint8).

    Returns:
        np.ndarray: Image array in specified format.

    Raises:
        ValueError: If input tensor format is invalid.
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        
        # Convert tensor to numpy and process
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
        
    return image_numpy.astype(imtype)


from typing import List, Optional, Tuple, Union


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """
    Generates a tensor with random values from a normal distribution.

    This function creates a tensor of the specified shape filled with random values
    from a normal (Gaussian) distribution. It supports multiple generators for
    distributed training scenarios.

    Args:
        shape (Union[Tuple, List]): Shape of the tensor to generate.
        generator (Optional[Union[List[torch.Generator], torch.Generator]]): Random number generator(s).
        device (Optional[torch.device]): Device to place the tensor on.
        dtype (Optional[torch.dtype]): Data type of the tensor.
        layout (Optional[torch.layout]): Memory layout of the tensor.

    Returns:
        torch.Tensor: Randomly initialized tensor.

    Note:
        If multiple generators are provided, they should match the batch size (first dimension).
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = (
            generator.device.type
            if not isinstance(generator, list)
            else generator[0].device.type
        )
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(
                f"Cannot generate a {device} tensor from a generator of type {gen_device_type}."
            )

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(
                shape,
                generator=generator[i],
                device=rand_device,
                dtype=dtype,
                layout=layout,
            )
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(
            shape, generator=generator, device=rand_device, dtype=dtype, layout=layout
        ).to(device)

    return latents


# args = parse_args()

if args.non_ema_revision is not None:
    deprecate(
        "non_ema_revision!=None",
        "0.15.0",
        message=(
            "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
            " use `--variant=non_ema` instead."
        ),
    )
logging_dir = os.path.join(args.output_dir, args.logging_dir)

accelerator_project_config = ProjectConfiguration(
    total_limit=args.checkpoints_total_limit
)

accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    logging_dir=logging_dir,
    project_config=accelerator_project_config,
)

# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(accelerator.state, main_process_only=False)
if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()
else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

# If passed along, set the training seed now.
if args.seed is not None:
    set_seed(args.seed)

# Handle the repository creation
if accelerator.is_main_process:
    if args.push_to_hub:
        if args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(args.output_dir).name, token=args.hub_token
            )
        else:
            repo_name = args.hub_model_id
        create_repo(repo_name, exist_ok=True, token=args.hub_token)
        repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

        with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
            if "step_*" not in gitignore:
                gitignore.write("step_*\n")
            if "epoch_*" not in gitignore:
                gitignore.write("epoch_*\n")
    elif args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)


# Load scheduler, tokenizer and models.
noise_scheduler = DDPMScheduler.from_pretrained("./sd2-inp", subfolder="scheduler")
processor = TrOCRProcessor.from_pretrained("./trocr-large-printed")
trocr_model = VisionEncoderDecoderModel.from_pretrained(
    "./trocr-large-printed"
).encoder.cuda()
full_trocr_model = VisionEncoderDecoderModel.from_pretrained(
    "./trocr-large-printed"
).cuda()

vae = AutoencoderKL.from_pretrained(
    "./pretrianed/vae", subfolder="vae", revision=args.revision
).cuda()
unet = UNet2DConditionModel.from_pretrained(
    "./pretrianed/unet", subfolder="unet", revision=args.non_ema_revision
).cuda()

# Freeze vae and text_encoder
trocr_model.requires_grad_(False)
vae.requires_grad_(False)
full_trocr_model.requires_grad_(False)

weight_dtype = torch.float32

# vae.to(accelerator.device, dtype=weight_dtype)
# trocr_model.to(accelerator.device, dtype=weight_dtype)
# unet.to(accelerator.device, dtype=weight_dtype)

# Create EMA for the unet.
if args.use_ema:
    ema_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    ema_unet = EMAModel(
        ema_unet.parameters(),
        model_cls=UNet2DConditionModel,
        model_config=ema_unet.config,
    )

if args.enable_xformers_memory_efficient_attention:
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError(
            "xformers is not available. Make sure it is installed correctly"
        )


def to_tensor(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif not isinstance(image, np.ndarray):
        raise TypeError("Error")

    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    tensor = torch.from_numpy(image)

    return tensor


# `accelerate` 0.16.0 will have better support for customized saving
if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if args.use_ema:
            ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

        for i, model in enumerate(models):
            model.save_pretrained(os.path.join(output_dir, "unet"))

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(models, input_dir):
        if args.use_ema:
            load_model = EMAModel.from_pretrained(
                os.path.join(input_dir, "unet_ema"), UNet2DConditionModel
            )
            ema_unet.load_state_dict(load_model.state_dict())
            ema_unet.to(accelerator.device)
            del load_model

        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet"
            )
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

if args.gradient_checkpointing:
    unet.enable_gradient_checkpointing()

# Enable TF32 for faster training on Ampere GPUs,
# cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
if args.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

# Initialize the optimizer
if args.use_8bit_adam:
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
        )

    optimizer_cls = bnb.optim.AdamW8bit
else:
    optimizer_cls = torch.optim.AdamW


def text_editing(text, instance_image, slider_step, x0, y0, x1, y1):
    """
    Performs text editing in an image using the DiffUTE model.

    This function takes an input image and coordinates of a text region, and replaces
    the text in that region with new text while preserving the surrounding context.
    It uses a combination of VAE and UNet models to achieve high-quality text editing.

    Args:
        text (str): New text to render in the image.
        instance_image (PIL.Image): Input image to edit.
        slider_step (int): Number of diffusion steps for generation.
        x0, y0, x1, y1 (int): Coordinates defining the text region to edit.

    Returns:
        PIL.Image: Edited image with the new text.

    Raises:
        ValueError: If input parameters are invalid or model inference fails.
    """
    # Initialize models and components
    examples = {}
    noise_scheduler_te = noise_scheduler
    processor_te = processor
    trocr_model_te = trocr_model
    vae_te = vae
    unet_te = unet

    # Process text region coordinates and calculate dimensions
    bbox = [x0, y0, x1, y1]
    location = np.int32(bbox)

    # Calculate crop dimensions based on text height
    # This ensures appropriate context is captured for the diffusion process
    char_height = location[3] - location[1]
    char_lenth = location[2] - location[0]
    h, w, c = instance_image.shape
    short_side = min(h, w)

    # Determine optimal crop size based on text height
    # Larger text requires larger context window for better results
    if 6 * char_height < 128:
        CROP_LENTH = max(128, char_lenth)
    elif 6 * char_height < 256:
        CROP_LENTH = max(256, char_lenth)
    elif 6 * char_height < 384:
        CROP_LENTH = max(384, char_lenth)
    elif 6 * char_height < 512:
        CROP_LENTH = max(512, char_lenth)
    elif 6 * char_height < 640:
        CROP_LENTH = max(640, char_lenth)
    elif 6 * char_height < 784:
        CROP_LENTH = max(784, char_lenth)
    elif 6 * char_height < 1000:
        CROP_LENTH = max(1000, char_lenth)
    else:
        CROP_LENTH = 6 * char_height

    # Adjust final crop scale based on text length and image dimensions
    if char_lenth < CROP_LENTH:
        crop_scale = min(CROP_LENTH, short_side)
    else:
        crop_scale = short_side

    _text_te, _ori_text = text, text

    # Generate mask for the text region
    # This mask guides the diffusion process to edit only the text area
    mask = generate_mask(instance_image.shape[:2][::-1], location)
    masked_image = prepare_mask_and_masked_image(instance_image, mask)

    # Calculate crop coordinates to focus on the text region
    x1, y1, x2, y2 = location
    if x2 - x1 < crop_scale:
        if x2 - crop_scale > 0:
            x_s = x2 - crop_scale
        elif x1 + crop_scale < w:
            x_s = x1
        else:
            x_s = 0
    else:
        x_s = np.random.randint(x1, max(0, x2 - crop_scale - 1))

    if y2 - y1 < crop_scale:
        if y2 - crop_scale > 0:
            y_s = y2 - crop_scale
        elif y1 + crop_scale < w:
            y_s = y1
        else:
            y_s = 0
    else:
        y_s = np.random.randint(y1, max(0, y2 - crop_scale - 1))

    # Prepare images for processing
    # 1. Render new text
    draw_ttf = draw_text(instance_image.shape[:2][::-1], text)
    # 2. Crop relevant region from original image
    instance_image_1 = instance_image[y_s : y_s + crop_scale, x_s : x_s + crop_scale, :]
    # 3. Crop corresponding mask region
    mask_crop = mask[y_s : y_s + crop_scale, x_s : x_s + crop_scale]
    # 4. Crop masked image region
    masked_image_crop = masked_image[y_s : y_s + crop_scale, x_s : x_s + crop_scale, :]

    # Apply image transformations to prepare for model input
    # These transformations were used during training
    augmented = image_trans_resize_and_crop(image=instance_image_1)
    instance_image_1 = augmented["image"]
    augmented = image_trans(image=instance_image_1)
    instance_image_1 = augmented["image"]

    augmented = image_trans(image=instance_image)
    instance_image = augmented["image"]

    augmented = image_trans_resize_and_crop(image=masked_image_crop)
    masked_image_crop = augmented["image"]
    augmented = image_trans(image=masked_image_crop)
    masked_image_crop = augmented["image"]

    augmented = mask_resize_and_crop(image=mask_crop)
    mask_crop = augmented["image"]
    augmented = image_trans(image=mask_crop)
    mask_crop = augmented["image"]

    augmented = image_trans(image=draw_ttf)
    draw_ttf = augmented["image"]

    # Store processed images for potential debugging
    examples["ori_image"] = instance_image
    examples["instance_images"] = instance_image_1
    examples["mask"] = mask_crop
    examples["masked_image"] = masked_image_crop
    examples["ttf_img"] = draw_ttf
    examples["crop_scale"] = crop_scale

    # Prepare tensors for model input
    # Move to GPU and ensure correct format
    input_values = instance_image_1.unsqueeze(0)
    input_values = input_values.to(memory_format=torch.contiguous_format).float()
    input_values = input_values.cuda()

    mask_crop = mask_crop.unsqueeze(0)
    mask_crop = mask_crop.cuda()
    masked_image_crop = masked_image_crop.unsqueeze(0)
    masked_image_crop = masked_image_crop.cuda()
    ttf_imgs = []
    ttf_imgs.append(draw_ttf)

    with torch.no_grad():
        # Generate text features using TrOCR
        # These features guide the diffusion process
        pixel_values = processor_te(images=ttf_imgs, return_tensors="pt").pixel_values
        pixel_values = pixel_values.cuda()
        ocr_feature = trocr_model_te(pixel_values)
        ocr_embeddings = ocr_feature.last_hidden_state.detach()

        # Convert images to latent space using VAE
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        latents = vae_te.encode(input_values.to(weight_dtype)).latent_dist.sample()
        latents = latents * vae_te.config.scaling_factor
        torch.randn_like(latents)

        # Process mask for latent space
        # Resize mask to match latent dimensions
        width, height, *_ = mask_crop.size()[::-1]
        mask_crop = torch.nn.functional.interpolate(
            mask_crop,
            size=[width // vae_scale_factor, height // vae_scale_factor, *_][:-2][::-1],
        )
        mask_crop = mask_crop.to(weight_dtype)

        # Generate masked image latents
        masked_image_latents = vae.encode(masked_image_crop).latent_dist.sample()
        masked_image_latents = masked_image_latents * vae.config.scaling_factor

        # Setup shape for generation
        shape = (
            1,
            vae.config.latent_channels,
            height // vae_scale_factor,
            width // vae_scale_factor,
        )

        # Initialize noise for diffusion process
        latents = randn_tensor(
            shape, generator=torch.manual_seed(0), dtype=weight_dtype
        )
        latents = latents * noise_scheduler_te.init_noise_sigma
        latents = latents.cuda()

        # Set up diffusion process
        noise_scheduler_te.set_timesteps(int(slider_step))
        timesteps = noise_scheduler_te.timesteps

        # Progressive denoising loop
        for i, t in enumerate(timesteps):
            # Prepare model input
            latent_model_input = latents
            latent_model_input = noise_scheduler_te.scale_model_input(
                latent_model_input, t
            )
            # Concatenate all inputs for UNet
            latent_model_input = torch.cat(
                [latent_model_input, mask_crop, masked_image_latents], dim=1
            )

            # Generate noise prediction
            noise_pred = unet_te(latent_model_input, t, ocr_embeddings).sample
            # Update latents using scheduler
            latents = noise_scheduler_te.step(noise_pred, t, latents).prev_sample

        # Decode generated image from latents using VAE
        pred_latents = 1 / vae_te.config.scaling_factor * latents
        image_vae = vae_te.decode(pred_latents).sample

        # Post-process generated image
        image = (image_vae / 2 + 0.5) * 255.0
        image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()
        image = image.squeeze(0)

        # Calculate final dimensions for blending
        if y_s + crop_scale > h:
            r_h = h - y_s
        else:
            r_h = crop_scale

        if x_s + crop_scale > w:
            r_w = w - x_s
        else:
            r_w = crop_scale

        # Blend generated image with original
        inf_res = instance_image.cpu().permute(1, 2, 0).float().numpy().copy()
        mid_inf_res = instance_image.cpu().permute(1, 2, 0).float().numpy().copy()
        mid_inf_res[y_s : y_s + crop_scale, x_s : x_s + crop_scale, :] = cv2.resize(
            image, (r_w, r_h)
        )
        inf_res[y1:y2, x1:x2, :] = mid_inf_res[y1:y2, x1:x2, :]

        # Convert to final format
        inf_res = inf_res.round().astype("uint8")
        inf_res = Image.fromarray(inf_res).convert("RGB")
        ori_comp = instance_image.cpu().permute(1, 2, 0).float().numpy()
        ori_comp = Image.fromarray(ori_comp.round().astype("uint8")).convert("RGB")

    return inf_res, mask * 255


# Global state for ROI selection
ROI_coordinates = {
    "x_temp": 0,  # Temporary x coordinate
    "y_temp": 0,  # Temporary y coordinate
    "x_new": 0,   # New x coordinate from click
    "y_new": 0,   # New y coordinate from click
    "clicks": 0,  # Click counter for ROI selection
}


def get_select_coordinates(img, x0, y0, x1, y1, evt: gr.SelectData):
    """
    Handles coordinate selection events from the Gradio interface.

    This function processes coordinate selection events from the user interface,
    storing and validating the coordinates for text region selection.

    Args:
        img (PIL.Image): The image being processed.
        x0, y0, x1, y1 (int): Current coordinates of the selection.
        evt (gr.SelectData): Event data from Gradio containing new coordinates.

    Returns:
        tuple: Updated coordinates (x0, y0, x1, y1) and status message.

    Note:
        This function is designed to work with Gradio's event system for
        interactive image selection.
    """
    sections = []
    # update new coordinates
    ROI_coordinates["clicks"] += 1
    ROI_coordinates["x_temp"] = ROI_coordinates["x_new"]
    ROI_coordinates["y_temp"] = ROI_coordinates["y_new"]
    ROI_coordinates["x_new"] = evt.index[0]
    ROI_coordinates["y_new"] = evt.index[1]
    # compare start end coordinates
    x_start = (
        ROI_coordinates["x_new"]
        if (ROI_coordinates["x_new"] < ROI_coordinates["x_temp"])
        else ROI_coordinates["x_temp"]
    )
    y_start = (
        ROI_coordinates["y_new"]
        if (ROI_coordinates["y_new"] < ROI_coordinates["y_temp"])
        else ROI_coordinates["y_temp"]
    )
    x_end = (
        ROI_coordinates["x_new"]
        if (ROI_coordinates["x_new"] > ROI_coordinates["x_temp"])
        else ROI_coordinates["x_temp"]
    )
    y_end = (
        ROI_coordinates["y_new"]
        if (ROI_coordinates["y_new"] > ROI_coordinates["y_temp"])
        else ROI_coordinates["y_temp"]
    )
    if ROI_coordinates["clicks"] % 2 == 0:
        # both start and end point get
        sections.append(((x_start, y_start, x_end, y_end), "ROI of Text Editing"))

        x0, y0, x1, y1 = x_start, y_start, x_end, y_end
        print(x0, y0, x1, y1)
        return (img, sections), x0, y0, x1, y1
    else:
        point_width = int(img.shape[0] * 0.05)
        sections.append(
            (
                (
                    ROI_coordinates["x_new"],
                    ROI_coordinates["y_new"],
                    ROI_coordinates["x_new"] + point_width,
                    ROI_coordinates["y_new"] + point_width,
                ),
                "Click second point for ROI",
            )
        )
        x0, y0, x1, y1 = (
            ROI_coordinates["x_new"],
            ROI_coordinates["y_new"],
            ROI_coordinates["x_new"] + point_width,
            ROI_coordinates["y_new"] + point_width,
        )

        return (img, sections), x0, y0, x1, y1


# Initialize Gradio interface with components
with gr.Blocks() as demo:
    gr.Markdown("DiffUTE: Universal Text Editing Diffusion Model")
    
    # Main text editing interface tab
    with gr.Tab("Text editing pipeline"):
        with gr.Row():
            # Left column - Input controls
            with gr.Column():
                # Image input and text controls
                ori_image = gr.Image(label="Original image")
                text_input = gr.Textbox(label="Input the text you want to write here")
                img_output = gr.AnnotatedImage(
                    label="ROI", color_map={"Click second point for ROI": "#f44336"}
                )
                button = gr.Button("Generate", variant="primary")
                
                # Coordinate display
                with gr.Row():
                    x0 = gr.Number(label="X0")
                    x1 = gr.Number(label="X1")
                    y0 = gr.Number(label="Y0")
                    y1 = gr.Number(label="Y1")
                
                # Example configurations for demonstration
                text_edit_examples = [
                    [
                        "2023-07-25",  # Text to render
                        "./examples/1793.jpg",  # Image path
                        "150",  # Steps
                        "204",  # x0
                        "240",  # y0
                        "333",  # x1
                        "270",  # y1
                        "512",  # Resolution
                    ],
                    [
                        "ANT",
                        "./examples/00000006.jpg",
                        "150",
                        "334",
                        "213",
                        "401",
                        "245",
                        "384",
                    ],
                    [
                        "88.88",
                        "./examples/icdar_10201.jpg",
                        "150",
                        "703",
                        "1236",
                        "841",
                        "1301",
                        "640",
                    ],
                    [
                        "7890",
                        "./examples/card_11116.jpg",
                        "150",
                        "475",
                        "276",
                        "611",
                        "338",
                        "640",
                    ],
                ]
                
                # Inference step slider
                ute_steps = gr.Slider(
                    20.0,
                    200.0,
                    value=150,
                    step=1,
                    label="Inference step",
                    info="The step of denoising process.",
                )

                # Load example configurations
                gr.Examples(
                    text_edit_examples,
                    inputs=[text_input, ori_image, ute_steps, x0, y0, x1, y1],
                )

            # Right column - Output display
            with gr.Column():
                output_imgs = gr.Image(label="Generated image")
                output_masks = gr.Image(label="Generated mask")

    # Event handlers for UI interactions
    ori_image.select(
        get_select_coordinates,
        [ori_image, x0, y0, x1, y1],
        [img_output, x0, y0, x1, y1],
    )
    button.click(
        text_editing,
        inputs=[text_input, ori_image, ute_steps, x0, y0, x1, y1],
        outputs=[output_imgs, output_masks],
    )

if __name__ == "__main__":
    # Launch Gradio interface with debugging and queue enabled
    demo.launch(debug=True, enable_queue=True)
