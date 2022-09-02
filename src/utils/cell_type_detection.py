# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn import mixture
import scipy.spatial as ss
from sklearn.neighbors import NearestNeighbors


def get_positive_cells_batch(dataset, img_names):

    postive_ids = []

    for i in range(len(img_names)):
        img_subset = dataset[dataset["image"] == img_names[i]]
        postive_ids.extend(get_positive_cells(img_subset))

    return postive_ids


def get_positive_cells(dataset, feature="int_mean", id_to_return="nuc_id"):

    dat = np.array(dataset[feature]).reshape(-1, 1)

    positive_cells = dataset[assign_cell_status(dat)][id_to_return].tolist()

    return positive_cells


def assign_cell_status(dat):

    thresh = get_two_component_threshold(dat)

    status = dat > thresh

    return status


def get_two_component_threshold(data):

    gm = mixture.GaussianMixture(n_components=2, random_state=0).fit(data)
    threshold = np.mean(gm.means_)

    return threshold


def tcell_k_neighbors(
    dataset, k=5, tcell_label="t_cells", cell_type_col="type", id_col="nuc_id"
):
    image_ids = dataset["image"].unique()
    grouped = dataset.groupby(dataset.image)
    nn = NearestNeighbors(n_neighbors=k, p=2, n_jobs=8)
    tcell_kneighbors = []
    for i in range(len(image_ids)):
        data = grouped.get_group(image_ids[i])
        cell_type_labels = np.array(data.loc[:, cell_type_col])
        ids = np.array(data.loc[:, id_col])
        nn.fit(data.loc[:, ["centroid_y", "centroid_x"]])
        knbrs = nn.kneighbors(
            data.loc[:, ["centroid_y", "centroid_x"]],
            n_neighbors=k + 1,
            return_distance=False,
        )

        for i in range(len(knbrs)):
            if tcell_label in cell_type_labels[knbrs[i]]:
                tcell_kneighbors.append(ids[i])
    return tcell_kneighbors


def t_cell_neighbours(dataset, R=1, tcell_label="t_cells", cell_type_col="type"):

    image_ids = dataset["image"].unique()
    grouped = dataset.groupby(dataset.image)

    tcell_neighbours = []

    for i in range(len(image_ids)):
        data = grouped.get_group(image_ids[i])
        t_cells = data.loc[data.loc[:, cell_type_col] == tcell_label, "nuc_id"]
        cords = np.column_stack((data["centroid_y"], data["centroid_x"]))
        # obtain the distance matrix
        dist_matrix = ss.distance.squareform(ss.distance.pdist(cords, "euclidean"))
        dist_matrix = pd.DataFrame(dist_matrix)
        dist_matrix.columns = data["nuc_id"]

        # Defining neighbourhood radius "R" and counting the number of nuclei in "R"
        t_cell_neighbours = dist_matrix[dist_matrix.columns.intersection(t_cells)]
        mask = ((t_cell_neighbours < R) & (t_cell_neighbours > 0)).astype(float)
        status = np.nansum(mask, axis=1)

        tcell_neighbours.extend(data["nuc_id"][status >= 1])

    return tcell_neighbours
