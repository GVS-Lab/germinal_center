# -*- coding: utf-8 -*-
from glob import glob
from pathlib import Path

from PIL import Image
from skimage import segmentation
from tifffile import imread
from tqdm import tqdm


def cell_seg_dilation_batch(labelled_image_path: str, output_dir: str, rad=10):
    """
    Function that reads in the segmented/labelled images for a field of view and computes cell boundary features.
    Args:
        labelled_image_path: path pointing to the segmented image
        output_dir: path where the results need to be stored
        rad: radius of dilation
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    seg_image_dirs = sorted(glob(labelled_image_path + "*.tif"))

    for i in tqdm(range(len(seg_image_dirs))):
        labelled_image = imread(seg_image_dirs[i])
        cell_boundaries = segmentation.expand_labels(labelled_image, distance=rad)
        im = Image.fromarray(cell_boundaries)
        im.save(output_dir + "/" + seg_image_dirs[i].rsplit("/", 1)[-1][:-4] + ".tif")
