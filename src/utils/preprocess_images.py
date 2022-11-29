# -*- coding: utf-8 -*-
import os.path
from pathlib import Path

import imageio as imio
import numpy as np
from tifffile import imsave

from src.utils.base import get_file_list


def extract_channel_save_image(image_dir: str, output_dir: str, channel: int):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # read in the images,segment and save labels
    img_locs = get_file_list(image_dir)

    for i in range(len(img_locs)):
        image_loc = img_locs[i]
        X = imio.imread(image_loc)  # read
        X = X[:, :, -channel]  # extract channel
        imsave(os.path.join(output_dir, os.path.split(image_loc)[1]), X)  # save


def quantile_normalize_and_save_images(
    image_dir: str, output_dir: str, mask_dir: str = None, quantiles=[0.01, 0.998]
):
    os.makedirs(output_dir, exist_ok=True)
    image_locs = get_file_list(image_dir)
    for loc in image_locs:
        img = imio.imread(loc)
        if mask_dir is not None:
            mask = imio.imread(os.path.join(mask_dir, os.path.split(loc)[1]))
            masked_img = np.ma.array(img, mask=~mask).astype(float)
            low = np.quantile(masked_img, quantiles[0])
            high = np.quantile(masked_img, quantiles[1])
        else:
            low = np.quantile(img, quantiles[0])
            high = np.quantile(img, quantiles[1])
        scaled_img = (img - low) / (high - low)
        scaled_img = np.clip(scaled_img * 255, 0, 255).astype(np.uint8)
        imsave(os.path.join(output_dir, os.path.split(loc)[1]), scaled_img)
