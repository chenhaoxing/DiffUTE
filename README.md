# DiffUTE
This repository is the code of our paper "DiffUTE: Universal Text Editing Diffusion Model". Unfortunately, pre-trained models are not allowed to be made public due to the lisence of AntGroup.

## Getting Started with DiffUTE
### Installation
The codebases are built on top of [diffusers](https://github.com/huggingface/diffusers). Thanks very much.

#### Requirements
- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 1.10.0 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV
- transformers
#### Steps
1. Install diffusers following https://github.com/huggingface/diffusers.

2. Prepare datasets. Due to data sensitivity issues, our data will not be publicly available now, you can reproduce it on your own data, and all images with text are available for model training. Because our data is present on [Ali-Yun oss](https://www.aliyun.com/search?spm=5176.22772544.J_8058803260.37.4aa92ea9DAomsC&k=OSS&__is_mobile__=false&__is_spider__=false&__is_grey__=false), we have chosen pcache to read the data we have stored. You can change the data reading method according to the way you store the data.

3. Train VAE
```bash
export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_vae.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=6 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --learning_rate=5e-6 \
  --num_train_epochs=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=3000 \
```
4. Train DiffUTE
```bash
export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_diffute.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=32 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=5000 \
  --num_train_epochs=5 \
  --checkpointing_steps=10000
```

## Citing DiffUTE

If you use DiffUTE in your research or wish to refer to the baseline results published here, please use the following BibTeX entry.

```BibTeX
@article{DiffUTE,
      title={DiffUTE: Universal Text Editing Diffusion Model},
      author={Chen, Haoxing and Xu, Zhuoer and Gu, Zhangxuan and Lan, Jun and Zheng, Xing and Li, Yaohui and Meng, Changhua and Zhu, Huijia and Wang, Weiqiang},
      journal={arXiv preprint arXiv:2305.10825},
      year={2023}
}
```
## Acknowledgement
Many thanks to the nice work of [diffusers](https://github.com/huggingface/diffusers).

## Contacts
Please feel free to contact us if you have any problems.

Email: [hx.chen@hotmail.com](hx.chen@hotmail.com) or [zhuoerxu.xzr@antgroup.com](zhuoerxu.xzr@antgroup.com)
