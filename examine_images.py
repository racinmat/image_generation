from PIL import Image
import numpy as np
img1 = r'D:\Projects\image_generation\animefest_selected_bench_small\00007-2797759059.png'
img2 = r'D:\Projects\image_generation\stable-diffusion-webui\outputs\txt2img-images\2023-05-13\00054-2797759059.png'
im1 = Image.open(img1)
im2 = Image.open(img2)
arr1 = np.array(im1)
arr2 = np.array(im2)
print('done')