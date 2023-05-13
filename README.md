# A playground for image generation stuff

The other stuff are here as submodules:
```shell
git submodule add git@github.com:racinmat/stable-diffusion-webui.git stable-diffusion-webui
```

I set up the new conda env for the stable diffusion webui like this:
```shell
conda create -n image_generation python==3.10
conda activate image_generation
```
I run the ui as
```shell
cd stable-diffusion-webui
webui-user.bat
```

The tutorial for getting started is: https://rentry.org/voldy#-guide-

## My notes on the calibration

My calibrations on 1080Ti with flags `--theme dark`.

the asuka after calibration, Euler setting:
![image](notable_outputs/00003-2870305590.png)
the transition on the left leg is too sharp, and with Euler A, there are some different artifacts on the hair,
not mentioned in the guide https://imgur.com/a/s3llTE5, see ![image](notable_outputs/00008-2870305590.png).

Even the baseline Asuka in https://imgur.com/a/DCYJCSX is not the same as the one in guide.
It seems that left leg artifact with gradient from pink to red is done by by the TensorCore, 
because that supports fp16 and not fp32 and the artifact with leg is same as if you run the Stable Diffusion with 
--no-half, which forces the VAE to be in fp32, while the default is fp16, I guess to accelerate the training and also 
utilize the tensor cores, so we probably see the difference between CUDA and TensorCore arithmetics.

So I assume this result is ok.

## Runtime notes:
Everything measured before and after generating Eurler Asuka. 

With default memory-related settings, after the start it takes 1.5GB ram idling, over 2GB ram after image is generated
and network loaded.

Using https://allthings.how/how-to-check-vram-usage-on-windows-10/ the task manager shows the python process takes 
2.8GB of GPU memory. It took 11s to generate Asuka (2.5it/s).
With `--medvram` it takes 2.6GB ram idling and after generating, it takes 4.2 GB RAM and 736MB of GPU memory.
It took 11s to generate Asuka (2.5it/s).
With `--lowram` it takes 2.7GB ram idling and after generating, it takes 4.2 GB RAM and 736MB of GPU memory.
Seems that it does not make any difference for our usecase.
It took 56s to generate Asuka (2.0s/it).
`--use-cpu all` without `--no-half` tails on https://huggingface.co/CompVis/stable-diffusion-v1-4/discussions/64
With `--use-cpu all --no-half` it takes 4.9GB ram idling and after generating, it takes 4.9GB RAM and 142MB of GPU memory.
Seems that it does not make any difference for our usecase.
It took 7m,3s to generate Asuka (15.14s/it).
It also yields completely different results (same as the cpu-only result in the imgur website).
It takes too long, completely unusable for our cases.

Some measurements are here https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Optimizations#memory--performance-impact-of-optimizers-and-flags

Analyzing required cuda/rocm: the webui installs packages from https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/ea9bd9fc7409109adcd61b897abc2c8881161256/requirements_versions.txt
it installs `pytorch_lightning==1.7.6`, which is https://github.com/Lightning-AI/lightning/blob/1.7.6/requirements/pytorch/base.txt
which wants `torch>=1.9.*, <1.13.0`. Based on the look at the venv dir, the torch 1.13.1+cu117 is installed.

Based on the https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html the cuda 11.7 requires drivers
`>=450.80.02` for linux and `>=452.39` for windows.
Based on https://tech.amikelive.com/node-930/cuda-compatibility-of-nvidia-display-gpu-drivers/ it needs GPU with 
compute capability from 3.5 to 8.6. List of compatible GPUs can be found here: https://developer.nvidia.com/cuda-gpus
from the relevant ones: RTX, or Txxxx, Quadro Pxxxx, or GeForce, which can be GTX 9xx and newer.

For AMD, rx 5700, RX 6000 series and rx 500 series work https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/5468
based on https://github.com/ROCm/ROCm.github.io/blob/master/hardware.md it seems lots of gpus are supported.

e.g. the https://www.howtogeek.com/853529/hardware-for-stable-diffusion/ overstates the minimal requirements.

some nice models
https://civitai.com/models/3850/kenshi
https://civitai.com/models/4384/dreamshaper
https://civitai.com/models/4201/realistic-vision-v13
https://civitai.com/models/3950/art-and-eros-aeros-a-tribute-to-beauty
https://civitai.com/models/5935/liberty
https://civitai.com/models/2661/uber-realistic-porn-merge-urpm
https://civitai.com/models/3627/protogen-v22-anime-official-release
https://civitai.com/models/4823/deliberate
https://civitai.com/models/1366/elldreths-lucid-mix
https://civitai.com/models/1259/elldreths-og-4060-mix
https://civitai.com/models/1274/dreamlike-diffusion-10
https://civitai.com/models/3666/protogen-x34-photorealism-official-release
https://civitai.com/models/5657/vinteprotogenmix-v10
https://civitai.com/models/1102/synthwavepunk

random notes:
anything v3 is based on novel ai https://www.youtube.com/watch?v=W4tzFI3e_xo&t=6s&ab_channel=Aitrepreneur

### notes about the models used
stuff from Anything-V3.0-fp16.zip -> Anything-V3.0
nai:
- novelaileak\stableckpt\animefull-final-pruned -> nai.ckpt and yaml
- novelaileak\stableckpt\animevae.pt -> nai.vae.pt

waifu diffusion vae downloaded from https://huggingface.co/hakurei/waifu-diffusion-v1-4/tree/main/vae
- don't use yaml
- kl-f8-anime2.ckpt -> wd-13.vae.pt
- wd-v1-3-float16.ckpt (from torrent) -> wd-13.ckpt 

other useful tutorial
https://rentry.org/hdgpromptassist#terms

## Debugging and tuning materials after AF

There have been 1814 images from AF, I selected a subset to cover
- borderline porn
- shiny ballsy stuff
- weird and broken faces
- some nice images to make sure they stay nice

I want to:
- get rid of ballsy stuff in default
- strengthen push away from porn
- fix faces
- keep the nice stuff

to recreate the stuff:
copy images from dirs to some separate dire.
Run `python prepare_input_list.py` with the dir name.
Then use this file as input to `Batch from imagelist A`.
In order to make it work, batch count and batch size must be 1, otherwise generated more images.
It still takes lots of time to go through 100 images, better to make even smaller set.

Recreating 30 images takes from 0:19 to 0:24, so ~5 mins.
Images are not the same pixel-wise, but look almost the same, enough for benchmarks.
Probably numeric instability.

todo: in case of default prompt, remove the best shadow.

Nový negative prompt:
```
(nsfw), (sex), (porn), (penis), (nipples), (vagina), anal, futa, (pussy), no panties, vaginal, (nude), (naked), bdsm, violence, (cum), cum in pussy, cum in mouth, x-ray, ahegao, erection, pubic hair, censored, testicles, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, deformed, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, futa, long neck, username, watermark, signature, see-through
```
