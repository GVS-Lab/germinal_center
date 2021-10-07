# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import seaborn as sns
from tqdm import tqdm


def plot_feature_space(
    data: pd.DataFrame,
    labels: np.ndarray = None,
    mode: str = "tsne",
    figsize: Tuple[int] = [8, 6],
    title: str = "",
    alpha: float = 1,
):
    idx = data.index
    scaled = StandardScaler().fit_transform(data)
    if mode == "tsne":
        embs = TSNE().fit_transform(scaled)
    elif mode == "umap":
        embs = UMAP().fit_transform(scaled)
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
    )
    plt.title(title)
    plt.show()
    plt.close()
