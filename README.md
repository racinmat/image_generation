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