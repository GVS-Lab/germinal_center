# -*- coding: utf-8 -*-
import os
from glob import glob
from pathlib import Path

import pandas as pd
from nmco.nuclear_features.int_dist_features import measure_intensity_features
from nmco.utils.run_nuclear_feature_extraction import run_nuclear_chromatin_feat_ext
from skimage import measure
from tifffile import imread
from tqdm import tqdm


def extract_nmco_feats_batch(
    raw_image_path: str, labelled_image_path: str, output_dir: str
):
    """
    Function that reads in the raw and segmented/labelled images for a field of view and computes nuclear features.
    Note this has been used only for DAPI stained images
    Args:
        raw_image_path: path pointing to the raw image directory
        labelled_image_path: path pointing to the segmented image directory
        output_dir: path where the results need to be stored
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    raw_image_dirs = sorted(glob(raw_image_path + "*.tif"))
    seg_image_dirs = sorted(glob(labelled_image_path + "*.tif"))

    all_features = pd.DataFrame()

    for i in range(len(raw_image_dirs)):
        features = run_nuclear_chromatin_feat_ext(
            raw_image_dirs[i],
            seg_image_dirs[i],
            output_dir,
            normalize=True,
            save_output=True,
        )
        features["image"] = seg_image_dirs[i].rsplit("/", 1)[1][:-4]

        all_features = all_features.append(features)

    return all_features


def extract_spatial_coordinates_batch(labelled_image_path: str, output_dir: str):
    """
    Function that reads in the raw and segmented/labelled images for a field of view and computes global nuclear position.
    Note this has been used only for DAPI stained images
    Args:
        labelled_image_path: path pointing to the segmented image directory
        output_dir: path where the results need to be stored
    """

    seg_image_dirs = sorted(glob(labelled_image_path + "*.tif"))

    all_features = pd.DataFrame()

    for i in range(len(seg_image_dirs)):
        features = get_nuclear_global_cordinates(seg_image_dirs[i], output_dir)
        features["image"] = seg_image_dirs[i].rsplit("/", 1)[1][:-4]
        features["nuc_id"] = (
            features["image"].astype(str) + "_" + features["label"].astype(str)
        )
        all_features = all_features.append(features)

    return all_features


def measure_intensity_batch(
    labelled_image_path: str, protein_image_path: str, output_dir: str
):
    """
    Function that reads in the segmented/labelled images for a field of view and computes cell boundary features.
    Args:
        labelled_image_path: path pointing to the segmented image
        output_dir: path where the results need to be stored
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    seg_image_dirs = sorted(glob(labelled_image_path + "*.tif"))
    prot_image_dirs = sorted(glob(protein_image_path + "*.tif"))

    all_features = pd.DataFrame()

    for i in tqdm(range(len(seg_image_dirs))):

        labelled_image = imread(seg_image_dirs[i])
        protein_image = imread(prot_image_dirs[i])

        # Get features for the individual nuclei in the image
        props = measure.regionprops(labelled_image, protein_image)

        features = pd.DataFrame()
        # Measure scikit's built in features

        for f in range(len(props)):
            features = features.append(
                pd.concat(
                    [
                        pd.DataFrame([f + 1], columns=["label"]),
                        measure_intensity_features(
                            regionmask=props[f].image,
                            intensity=props[f].intensity_image,
                            measure_int_dist=True,
                            measure_hc_ec_ratios=False,
                        ),
                    ],
                    axis=1,
                ),
                ignore_index=True,
            )

        features["image"] = seg_image_dirs[i].rsplit("/", 1)[1][:-4]
        features.to_csv(
            output_dir + "/" + seg_image_dirs[i].rsplit("/", 1)[1][:-4] + ".csv"
        )

        all_features = all_features.append(features)

    return all_features


def get_nuclear_global_cordinates(labelled_image_path: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    labelled_image = imread(labelled_image_path)

    data = pd.DataFrame(
        measure.regionprops_table(labelled_image, properties=("label", "centroid"))
    )
    data.to_csv(os.path.join(output_dir, "spatial_cordiates.csv"))

    return data
