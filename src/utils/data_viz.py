# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.image import imsave
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tifffile import imread
from umap import UMAP

from src.utils.base import get_file_list
from src.utils.discrimination import add_kfold_predictions


def plot_feature_space(
    data: pd.DataFrame,
    labels: np.ndarray = None,
    mode: str = "tsne",
    figsize=[8, 6],
    title: str = "",
    alpha: float = 1,
    random_state=1234,
    s=10,
    palette=None,
):
    idx = data.index
    scaled = StandardScaler().fit_transform(data)
    if mode == "tsne":
        embs = TSNE(random_state == random_state).fit_transform(scaled)
    elif mode == "umap":
        embs = UMAP(random_state=random_state).fit_transform(scaled)
    else:
        raise NotImplementedError("Unknown reduction type {}".format(mode))
    embs = pd.DataFrame(
        embs, columns=["{}_0".format(mode), "{}_1".format(mode)], index=idx
    )
    if labels is not None:
        embs["label"] = labels
        label_col = "label"
    else:
        label_col = None
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(
        data=embs,
        x="{}_0".format(mode),
        y="{}_1".format(mode),
        hue=label_col,
        alpha=alpha,
        s=s,
        palette=palette,
    )
    plt.title(title)
    plt.show()
    plt.close()

    return embs


def visualize_segmentation_results(
    image_dir, mask_dir, overlay_output_dir, outline_output_dir, alpha=0.3
):
    os.makedirs(overlay_output_dir, exist_ok=True)
    os.makedirs(outline_output_dir, exist_ok=True)
    image_locs = get_file_list(image_dir)
    for loc in image_locs:
        image = imread(loc)
        file_name = os.path.split(loc)[1]
        mask = imread(os.path.join(mask_dir, file_name))
        overlay_img = label2rgb(label=mask, image=image, bg_label=0, alpha=alpha)
        outline_img = mark_boundaries(label_img=mask, image=image, background_label=0)
        imsave(
            os.path.join(overlay_output_dir, file_name.split(".")[0] + ".png"),
            overlay_img,
        )
        imsave(
            os.path.join(outline_output_dir, file_name.split(".")[0] + ".png"),
            outline_img,
        )


def vis_classes(predictions, nuclear_data, path_to_raw_images, image_id):
    img_path = os.path.join(
        os.path.join(path_to_raw_images, "images"), str(image_id) + ".tif"
    )
    image = imread(img_path)
    colors = {
        "dark_b_cells": "red",
        "light_b_cells": "green",
        "t_cells": "blue",
        "none": "yellow",
    }
    grouped = nuclear_data.groupby(nuclear_data.image)
    data = grouped.get_group(image_id)
    x = nuclear_data.loc[data.index, "centroid_x"]
    y = nuclear_data.loc[data.index, "centroid_y"]
    pred_labels = predictions.loc[data.index, "predicted_class"]
    act_labels = predictions.loc[data.index, "actual_stage"]
    fig = plt.figure(figsize=(10, 4))
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)
    ax0.imshow(image, aspect="auto", origin="lower")
    ax1.scatter(x, y, c=(pred_labels.map(colors)), s=1)
    ax1.set_title("Predicted Classes")
    ax2.scatter(x, y, c=(act_labels.map(colors)), s=1)
    ax2.set_title("Actual Classes")


#


def plot_tcell_labels(
    data,
    spatial_cord,
    image_dir,
    label_col,
    cell_type_label_col,
    tcell_label="tcell",
    image_id=3,
    figsize=[12, 4],
    size=5,
    block_channel=None,
    palette1=None,
    palette2=None,
):
    image_path = os.path.join(image_dir, str(image_id) + ".tif")
    image = imread(image_path)
    if block_channel is not None:
        image[:, :, block_channel] = 0
    data = data.loc[data.image == image_id]
    x = np.array(spatial_cord.loc[data.index, "centroid-1"])
    y = np.array(spatial_cord.loc[data.index, "centroid-0"])
    labels = np.array(data.loc[:, label_col])
    vis_df = pd.DataFrame(x, columns=["x"], index=data.index)
    vis_df["y"] = y
    vis_df["T-cell interaction"] = labels
    vis_df["T-cell"] = np.array(data.loc[:, cell_type_label_col] == tcell_label)

    fig, ax = plt.subplots(figsize=figsize, ncols=3, sharey=True)
    ax = ax.flatten()
    ax[0].imshow(image, aspect="auto", origin="lower")
    ax[1] = sns.scatterplot(
        data=vis_df,
        x="x",
        y="y",
        hue="T-cell",
        s=size,
        ax=ax[1],
        hue_order=np.unique(vis_df.loc[:, "T-cell"]),
        palette=palette1,
    )
    ax[2] = sns.scatterplot(
        data=vis_df,
        x="x",
        y="y",
        hue="T-cell interaction",
        s=size,
        ax=ax[2],
        hue_order=np.unique(labels),
        palette=palette2,
    )
    for i in range(3):
        ax[i].set_xlabel("")
        ax[i].set_ylabel("")
    return fig, ax


def plot_predictions(
    model,
    data,
    spatial_cord,
    image_dir,
    selected_features,
    selected_labels,
    label_col,
    pos_label,
    image_id=0,
    figsize=[17, 4],
    n_folds=5,
    train_on_balanced_subsample=True,
    random_state=1234,
    s=5,
    palette1=None,
    palette2=None,
):
    figs = []
    axs = []
    img_path = os.path.join(image_dir, str(image_id) + ".tif")
    image = imread(img_path)
    data = data.loc[data.image == image_id]
    data = data.loc[data.loc[:, label_col].isin(selected_labels)]
    data = add_kfold_predictions(
        data=data,
        model=model,
        selected_features=selected_features,
        label_col=label_col,
        pos_label=pos_label,
        n_folds=n_folds,
        train_on_balanced_subsample=train_on_balanced_subsample,
        random_state=random_state,
    )
    x = np.array(spatial_cord.loc[data.index, "centroid-1"])
    y = np.array(spatial_cord.loc[data.index, "centroid-0"])
    preds = np.array(data.loc[:, "predicted"])
    pred_probs = np.array(data.loc[:, "predicted_{}_prob".format(pos_label)])
    labels = np.array(data.loc[:, label_col])

    pred_label_df = pd.DataFrame(x, columns=["x"])
    pred_label_df["y"] = y
    pred_label_df["predicted"] = preds
    pred_label_df["predicted {} probability".format(pos_label)] = pred_probs
    pred_label_df["label"] = np.array(labels)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, aspect="auto", origin="lower")
    figs.append(fig)
    axs.append(ax)

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(
        data=pred_label_df,
        x="x",
        y="y",
        hue="label",
        ax=ax,
        hue_order=np.unique(labels),
        s=s,
        palette=palette1,
    )
    sns.despine()
    ax.set_xlim([0, image.shape[1]])
    ax.set_ylim([0, image.shape[0]])
    figs.append(fig)
    axs.append(ax)

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(
        data=pred_label_df,
        x="x",
        y="y",
        hue="predicted",
        ax=ax,
        hue_order=np.unique(labels),
        s=s,
        palette=palette1,
    )
    sns.despine()
    ax.set_xlim([0, image.shape[1]])
    ax.set_ylim([0, image.shape[0]])
    figs.append(fig)
    axs.append(ax)

    fig, ax = plt.subplots(figsize=[figsize[0] + 1.5, figsize[1]])
    ax = sns.scatterplot(
        data=pred_label_df,
        x="x",
        y="y",
        hue="predicted {} probability".format(pos_label),
        palette=palette2,
        ax=ax,
        s=s,
    )
    sns.despine()
    ax.set_xlim([0, image.shape[1]])
    ax.set_ylim([0, image.shape[0]])

    norm = plt.Normalize(
        pred_label_df["predicted {} probability".format(pos_label)].min(),
        pred_label_df["predicted {} probability".format(pos_label)].max(),
    )
    sm = plt.cm.ScalarMappable(cmap=palette2, norm=norm)
    sm.set_array([])
    ax.get_legend().remove()
    ax.figure.colorbar(sm, label="{} prediction probability".format(pos_label))
    figs.append(fig)
    axs.append(ax)
    return figs, axs
