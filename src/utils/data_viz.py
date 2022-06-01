# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tifffile import imread
from umap import UMAP
import seaborn as sns
from tqdm import tqdm


def plot_feature_space(
    data: pd.DataFrame,
    labels: np.ndarray = None,
    mode: str = "tsne",
    figsize = [8, 6],
    title: str = "",
    alpha: float = 1,
    random_state=1234,
    s=10,
    palette=None
):
    idx = data.index
    scaled = StandardScaler().fit_transform(data)
    if mode == "tsne":
        embs = TSNE(random_state==random_state).fit_transform(scaled)
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


def vis_classes(predictions,  nuclear_data, path_to_raw_images, image_id):
    img_path = os.path.join(os.path.join(path_to_raw_images,"images"),str(image_id)+".tif")
    image = imread(img_path)
    colors = {'dark_b_cells':'red', 'light_b_cells':'green', 't_cells':'blue', 'none': 'yellow'}
    grouped = gc_nuc_features.groupby(gc_nuc_features.image)
    data = grouped.get_group(image_id)
    x = spatial_cord.loc[data.index,'centroid-0']
    y = spatial_cord.loc[data.index,'centroid-1']
    pred_labels = model_predictions.loc[data.index,"predicted_class"]
    act_labels = model_predictions.loc[data.index,"actual_stage"]
    fig = plt.figure(figsize=(10, 4))
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)
    ax0.imshow(image,aspect='auto',origin='lower') 
    ax1.scatter(y,x, c=(pred_labels.map(colors)),s=1)
    ax1.set_title('Predicted Classes')
    ax2.scatter(y,x, c=(act_labels.map(colors)),s=1)
    ax2.set_title('Actual Classes')
    