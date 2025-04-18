import gradio as gr
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFile

import argparse
import logging
from pathlib import Path
from typing import Optional
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
    font_size = 40
    font_file = "arialuni.ttf"
    len_text = len(text)
    if len_text == 0:
        len_text = 3
    img = Image.new("RGB", ((len_text + 2) * font_size, 60), color="white")
    # Define the font object
    font = ImageFont.truetype(font_file, font_size)
    # Define the text and position
    pos = (40, 10)

    draw = ImageDraw.Draw(img)
    draw.text(pos, text, font=font, fill="black")
    img = np.array(img)

    return img


def process_location(location, instance_image_size):
    h = location[3] - location[1]
    location[3] = min(location[3] + h / 10, instance_image_size[0] - 1)
    return location


def generate_mask(im_shape, ocr_locate):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(
        (ocr_locate[0], ocr_locate[1], ocr_locate[2], ocr_locate[3]),
        fill=1,
    )
    mask = np.array(mask)
    return mask


def prepare_mask_and_masked_image(image, mask):
    masked_image = np.multiply(
        image, np.stack([mask < 0.5, mask < 0.5, mask < 0.5]).transpose(1, 2, 0)
    )

    return masked_image


def download_oss_file_pcache(my_file="xxx"):
    # This function is no longer needed as we're not using OSS
    pass


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def tensor2im(input_image, imtype=np.uint8):
    """ "Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = (
            image_tensor[0].cpu().float().numpy()
        )  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (
            (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        )  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
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
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
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
    examples = {}
    noise_scheduler_te = noise_scheduler
    processor_te = processor
    trocr_model_te = trocr_model
    vae_te = vae
    unet_te = unet

    bbox = [x0, y0, x1, y1]

    location = np.int32(bbox)

    # crop scale对结果影响很大，可根据本批数据集进行微调
    char_height = location[3] - location[1]
    char_lenth = location[2] - location[0]

    h, w, c = instance_image.shape
    short_side = min(h, w)

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

    # 字体区域的长度 < 则按预设的crop值 > 则按规则取
    if char_lenth < CROP_LENTH:
        crop_scale = min(CROP_LENTH, short_side)
    else:
        crop_scale = short_side

    _text_te, _ori_text = text, text

    mask = generate_mask(instance_image.shape[:2][::-1], location)
    masked_image = prepare_mask_and_masked_image(instance_image, mask)
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

    draw_ttf = draw_text(instance_image.shape[:2][::-1], text)
    instance_image_1 = instance_image[y_s : y_s + crop_scale, x_s : x_s + crop_scale, :]
    mask_crop = mask[y_s : y_s + crop_scale, x_s : x_s + crop_scale]
    masked_image_crop = masked_image[y_s : y_s + crop_scale, x_s : x_s + crop_scale, :]

    # cv2.imwrite('ttf.jpg', draw_ttf)
    # cv2.imwrite('instance_image_1.jpg', instance_image_1)
    # cv2.imwrite('mask_crop.jpg', mask_crop*255)
    # cv2.imwrite('masked_image_crop.jpg', masked_image_crop)

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

    examples["ori_image"] = instance_image
    examples["instance_images"] = instance_image_1
    examples["mask"] = mask_crop
    examples["masked_image"] = masked_image_crop
    examples["ttf_img"] = draw_ttf
    examples["crop_scale"] = crop_scale

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
        pixel_values = processor_te(images=ttf_imgs, return_tensors="pt").pixel_values
        pixel_values = pixel_values.cuda()
        ocr_feature = trocr_model_te(pixel_values)
        ocr_embeddings = ocr_feature.last_hidden_state.detach()
        print(type(ocr_embeddings))

        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        # Convert images to latent space
        latents = vae_te.encode(input_values.to(weight_dtype)).latent_dist.sample()
        latents = latents * vae_te.config.scaling_factor
        torch.randn_like(latents)

        # Rex: prepare mask && mask latent as input of UNET
        width, height, *_ = mask_crop.size()[::-1]
        mask_crop = torch.nn.functional.interpolate(
            mask_crop,
            size=[width // vae_scale_factor, height // vae_scale_factor, *_][:-2][::-1],
        )
        mask_crop = mask_crop.to(weight_dtype)

        masked_image_latents = vae.encode(masked_image_crop).latent_dist.sample()
        masked_image_latents = masked_image_latents * vae.config.scaling_factor

        shape = (
            1,
            vae.config.latent_channels,
            height // vae_scale_factor,
            width // vae_scale_factor,
        )

        latents = randn_tensor(
            shape, generator=torch.manual_seed(0), dtype=weight_dtype
        )
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * noise_scheduler_te.init_noise_sigma
        latents = latents.cuda()

        noise_scheduler_te.set_timesteps(int(slider_step))
        timesteps = noise_scheduler_te.timesteps

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = latents
            # concat latents, mask, masked_image_latents in the channel dimension
            latent_model_input = noise_scheduler_te.scale_model_input(
                latent_model_input, t
            )
            latent_model_input = torch.cat(
                [latent_model_input, mask_crop, masked_image_latents], dim=1
            )

            # predict the noise residual
            noise_pred = unet_te(latent_model_input, t, ocr_embeddings).sample
            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler_te.step(noise_pred, t, latents).prev_sample
        # 11. Post-processing
        pred_latents = 1 / vae_te.config.scaling_factor * latents
        image_vae = vae_te.decode(pred_latents).sample

        # 13. Convert to PIL
        image = (image_vae / 2 + 0.5) * 255.0
        image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()
        image = image.squeeze(0)

        # h, w, _ = instance_image.shape
        if y_s + crop_scale > h:
            r_h = h - y_s
        else:
            r_h = crop_scale

        if x_s + crop_scale > w:
            r_w = w - x_s
        else:
            r_w = crop_scale

        inf_res = instance_image.cpu().permute(1, 2, 0).float().numpy().copy()
        mid_inf_res = instance_image.cpu().permute(1, 2, 0).float().numpy().copy()
        mid_inf_res[y_s : y_s + crop_scale, x_s : x_s + crop_scale, :] = cv2.resize(
            image, (r_w, r_h)
        )
        inf_res[y1:y2, x1:x2, :] = mid_inf_res[y1:y2, x1:x2, :]

        # generated_text_imgs = inf_res[y1:y2, x1:x2, :]
        # pixel_values = processor_te(images=generated_text_imgs, return_tensors="pt").pixel_values
        # pixel_values = pixel_values.cuda()
        # generated_ids = full_trocr_model_te.generate(pixel_values)
        # generated_text = processor_te.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(generated_text)
        inf_res = inf_res.round().astype("uint8")
        print(type(inf_res), type(mask))
        inf_res = Image.fromarray(inf_res).convert("RGB")
        ori_comp = instance_image.cpu().permute(1, 2, 0).float().numpy()
        ori_comp = Image.fromarray(ori_comp.round().astype("uint8")).convert("RGB")

    return inf_res, mask * 255


ROI_coordinates = {
    "x_temp": 0,
    "y_temp": 0,
    "x_new": 0,
    "y_new": 0,
    "clicks": 0,
}


def get_select_coordinates(img, x0, y0, x1, y1, evt: gr.SelectData):
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


with gr.Blocks() as demo:
    gr.Markdown("DiffUTE: Universal Text Editing Diffusion Model")
    with gr.Tab("Text editing pipeline"):
        with gr.Row():
            with gr.Column():
                ori_image = gr.Image(label="Original image")
                text_input = gr.Textbox(label="Input the text you want to write here")
                img_output = gr.AnnotatedImage(
                    label="ROI", color_map={"Click second point for ROI": "#f44336"}
                )
                button = gr.Button("Generate", variant="primary")
                with gr.Row():
                    x0 = gr.Number(label="X0")
                    x1 = gr.Number(label="X1")
                    y0 = gr.Number(label="Y0")
                    y1 = gr.Number(label="Y1")
                text_edit_examples = [
                    [
                        "2023-07-25",
                        "./examples/1793.jpg",
                        "150",
                        "204",
                        "240",
                        "333",
                        "270",
                        "512",
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
                ute_steps = gr.Slider(
                    20.0,
                    200.0,
                    value=150,
                    step=1,
                    label="Inference step",
                    info="The step of denoising process.",
                )

                gr.Examples(
                    text_edit_examples,
                    inputs=[text_input, ori_image, ute_steps, x0, y0, x1, y1],
                )

            with gr.Column():
                output_imgs = gr.Image(label="Generated image")
                output_masks = gr.Image(label="Generated mask")

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
    demo.launch(debug=True, enable_queue=True)
