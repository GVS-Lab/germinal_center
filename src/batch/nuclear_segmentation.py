# -*- coding: utf-8 -*-
"""
Contains function to segment object from images given a stardist model. 

"""
import imageio as imio
from glob import glob
from pathlib import Path
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from stardist import export_imagej_rois
from tifffile import imsave
import gc


def segment_objects_stardist2d(
    image_dir,
    output_dir_labels,
    output_dir_ijroi,
    use_pretrained=False,
    model_dir="models",
    model_name="DAPI_segmenation",
    prob_thresh=0.6,
    normalize_quants=[1, 99.8],
):

    Path(output_dir_labels).mkdir(parents=True, exist_ok=True)
    Path(output_dir_ijroi).mkdir(parents=True, exist_ok=True)

    # download / load a pretained model
    if use_pretrained:
        model = StarDist2D.from_pretrained("2D_versatile_fluo")
    else:
        model = StarDist2D(None, name=model_name, basedir=model_dir)
    # read in the images,segment and save labels
    all_images = sorted(glob(image_dir + "*.tif"))
    print(all_images)
    for i in range(len(all_images)):
        X = imio.imread(all_images[i])
        X = normalize(X, normalize_quants[0], normalize_quants[1], axis=(0, 1))
        labels, polygons = model.predict_instances(
            X, n_tiles=model._guess_n_tiles(X), prob_thresh=prob_thresh
        )
        imsave((output_dir_labels + all_images[i].rsplit("/", 1)[-1]), labels)
        if output_dir_ijroi:
            export_imagej_rois(
                (output_dir_ijroi + all_images[i].rsplit("/", 1)[-1][:-4] + ".zip"),
                polygons["coord"],
            )
        del X
        del labels
        del polygons
        gc.collect()
