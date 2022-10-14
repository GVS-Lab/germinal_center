# -*- coding: utf-8 -*-
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import stats, ranksums
from sklearn.metrics import plot_roc_curve, auc, confusion_matrix

from sklearn.model_selection import StratifiedKFold, cross_val_score, GroupKFold
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import fdrcorrection
import seaborn as sns
from tqdm import tqdm


def plot_conf_mtx(X, y, model, figsize=[6, 4], normalize_matrix="no"):
    if normalize_matrix == "yes":
        conf_mtx = confusion_matrix(y, model.predict(X), normalize="true")
    else:
        conf_mtx = confusion_matrix(y, model.predict(X))

    labels = np.array(y)

    conf_mtx = pd.DataFrame(
        conf_mtx, index=sorted(set(labels)), columns=sorted(set(labels))
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(conf_mtx, annot=True, cmap="BuPu", ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    plt.close()

    return fig


def compute_cv_scores(
    X, y, model, metrics=["accuracy", "balanced_accuracy", "f1_macro"], cv=10
):
    result = {"avg": [], "std": [], "min": [], "max": []}
    for metric in metrics:
        scores = cross_val_score(X=X, y=y, scoring=metric, cv=cv, estimator=model)
        result["avg"].append(np.mean(scores))
        result["std"].append(np.std(scores))
        result["min"].append(np.min(scores))
        result["max"].append(np.max(scores))
    result = pd.DataFrame(result)
    result.index = metrics
    return result


def plot_cv_conf_mtx(avg_conf_mtx, n_folds, figsize=[6, 4], annot_kws=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(
        avg_conf_mtx,
        annot=True,
        cmap="BuPu",
        ax=ax,
        vmin=0,
        vmax=1,
        annot_kws=annot_kws,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.title("Average confusion matrix from {}-fold stratified CV".format(n_folds))

    return fig, ax


def run_cv_evaluation(model, n_folds, features, labels, random_state=1234):
    models = []

    skf = StratifiedKFold(n_folds, shuffle=True, random_state=random_state)
    features = np.array(features)
    labels = np.array(labels)
    n_classes = len(np.unique(labels))

    confusion_mtx = np.zeros([n_classes, n_classes])
    for train_index, test_index in skf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        model.fit(X_train, y_train)
        confusion_mtx += confusion_matrix(
            y_test, model.predict(X_test), normalize="true"
        )
        models.append(copy.deepcopy(model))
    confusion_mtx = confusion_mtx / n_folds
    confusion_mtx = pd.DataFrame(
        confusion_mtx, index=sorted(set(labels)), columns=sorted(set(labels))
    )
    return models, confusion_mtx


def plot_feature_importance(
    importance,
    names,
    model_type,
    n_features=20,
    figsize=[6, 4],
    feature_color_dict=None,
):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)
    fi_df = fi_df.head(n_features)

    # Define size of bar plot
    fig = plt.figure(figsize=figsize)

    # Plot Searborn bar chart
    ax = sns.barplot(
        x=fi_df["feature_importance"], y=fi_df["feature_names"], color="dimgray"
    )
    sns.despine()

    if feature_color_dict is not None:
        for yticklabel in ax.get_yticklabels():
            yticklabel.set_color(feature_color_dict[yticklabel.get_text()])

    # Add chart labels

    plt.title(
        model_type
        + "Feature importance (top {} out of {} features)".format(
            n_features, len(feature_names)
        )
    )
    plt.xlabel("Feature importance")
    plt.ylabel("Features")
    plt.show()

    return fig


def find_markers(data, labels, test="welch"):
    results = []
    i = 0
    for label in tqdm(np.unique(labels), desc="Run marker screen"):
        label_results = {
            "label": [],
            "marker": [],
            "fc": [],
            "abs_delta_fc": [],
            "pval": [],
        }
        for c in data.columns:
            i += 1
            x = np.array(data.loc[labels == label, c])
            y = np.array(data.loc[labels != label, c])
            x = np.array(x[x != np.nan]).astype(float)
            y = np.array(y[y != np.nan]).astype(float)

            if test == "welch":
                pval = stats.ttest_ind(x, y, equal_var=False)[1]
            elif test == "ttest":
                pval = stats.ttest_ind(x, y, equal_var=False)[1]
            elif test == "wilcoxon":
                pval = ranksums(x, y)[1]
            else:
                raise NotImplementedError("Unknown test type: {}".format(test))
            fc = (np.mean(x) + 1e-15) / (np.mean(y) + 1e-15)
            label_results["label"].append(label)
            label_results["marker"].append(c)
            label_results["fc"].append(fc)
            label_results["abs_delta_fc"].append(abs(fc - 1))
            label_results["pval"].append(pval)
        label_result = pd.DataFrame(label_results)
        label_result.pval = label_result.pval.astype(float)
        label_result = label_result.sort_values("pval")
        results.append(label_result)
    result = pd.concat(results)
    result["adjusted_pval"] = fdrcorrection(np.array(result.loc[:, "pval"]))[1]
    return result.sort_values("adjusted_pval")


def plot_roc_for_stratified_cv(
    X,
    y,
    n_splits,
    classifier,
    title,
    pos_label=None,
    groups=None,
    random_state=1234,
    figsize=[8, 8],
):
    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        cv = GroupKFold(n_splits=n_splits)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=figsize)
    if groups is None:
        for i, (train, test) in enumerate(cv.split(X, y)):
            classifier.fit(X[train], y[train])
            viz = plot_roc_curve(
                classifier,
                X[test],
                y[test],
                name="ROC fold {}".format(i),
                alpha=0.3,
                lw=1,
                ax=ax,
                pos_label=pos_label,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
    else:
        for i, (train, test) in enumerate(cv.split(X, y, groups=groups)):
            classifier.fit(X[train], y[train])
            viz = plot_roc_curve(
                classifier,
                X[test],
                y[test],
                name="ROC fold {}".format(i),
                alpha=0.3,
                lw=1,
                ax=ax,
                pos_label=pos_label,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title)
    ax.legend(loc="lower right")
    return fig, ax, classifier


def tcell_radius_neighbors(
    dataset, radius=1, tcell_label="t_cells", cell_type_col="type", id_col="nuc_id"
):
    image_ids = dataset["image"].unique()
    grouped = dataset.groupby(dataset.image)
    nn = NearestNeighbors(radius=1, p=2, n_jobs=8)
    tcell_kneighbors = []
    for i in range(len(image_ids)):
        data = grouped.get_group(image_ids[i])
        cell_type_labels = np.array(data.loc[:, cell_type_col])
        ids = np.array(data.loc[:, id_col])
        nn.fit(data.loc[:, ["spat_centroid_y", "spat_centroid_x"]])
        knbrs = nn.radius_neighbors(
            data.loc[:, ["spat_centroid_y", "spat_centroid_x"]],
            radius=radius,
            return_distance=False,
        )
        for i in range(len(knbrs)):
            if tcell_label in cell_type_labels[knbrs[i]]:
                tcell_kneighbors.append(ids[i])
    return tcell_kneighbors


def add_predictions(model, data, selected_features, label_col, pos_label):
    preds = model.predict(data.loc[:, selected_features])
    pred_probs = model.predict_proba(data.loc[:, selected_features])
    data["{}_prediction".format(label_col)] = preds
    data["{}_prediction_prob".format(label_col)] = pred_probs[
        :, model.classes_ == pos_label
    ]
    data["{}_pos_label".format(label_col)] = pos_label
    return data


def add_kfold_predictions(
    data,
    model,
    selected_features,
    label_col,
    pos_label,
    n_folds=5,
    random_state=1234,
    train_on_balanced_subsample=True,
):
    features = data.loc[:, selected_features]
    labels = data.loc[:, label_col]
    ru = RandomUnderSampler(random_state=random_state)

    kfold = StratifiedKFold(n_splits=n_folds)
    for train_index, test_index in kfold.split(X=features, y=labels):
        train_features, train_labels = (
            features.iloc[train_index],
            labels.iloc[train_index],
        )
        if train_on_balanced_subsample:
            train_features, train_labels = ru.fit_resample(train_features, train_labels)
        model.fit(train_features, train_labels)
        preds = model.predict(features.iloc[test_index])
        pred_probs = model.predict_proba(features.iloc[test_index])
        data.loc[features.iloc[test_index].index, "predicted"] = preds
        data.loc[
            features.iloc[test_index].index, "predicted_{}_prob".format(pos_label)
        ] = pred_probs[:, model.classes_ == pos_label]
    return data


def get_distances_to_dz_lz_border(nuc_features, spatial_cord, alpha=0.02, n_jobs=5):
    nuc_data = nuc_features.loc[nuc_features.loc[:, "nuc_id"].isin(spatial_cord.index)]
    nuc_data.index = np.array(nuc_data.loc[:, "nuc_id"])
    spatial_data = spatial_cord.loc[nuc_features.index]

    dz_nuc_data = nuc_data.loc[nuc_data.loc[:, "cell_type"] == "DZ B-cells"]
    lz_nuc_data = nuc_data.loc[nuc_data.loc[:, "cell_type"] == "LZ B-cells"]

    dz_spatial_data = spatial_data.loc[dz_nuc_data.index]
    lz_spatial_data = spatial_data.loc[lz_nuc_data.index]

    result = spatial_data.loc[:, ["centroid-0", "centroid-1", "image"]]
    result.loc[:, "cell_type"] = nuc_data.loc[result.index, "cell_type"]
    nn = NearestNeighbors(n_neighbors=2, n_jobs=n_jobs)
    for image in np.unique(spatial_data.loc[:, "image"]):
        image_spatial_data = spatial_data.loc[spatial_data.loc[:, "image"] == image]
        k = int(len(image_spatial_data) * alpha)
        image_lz_spatial_data = lz_spatial_data.loc[
            lz_spatial_data.loc[:, "image"] == image
        ]
        image_dz_spatial_data = dz_spatial_data.loc[
            dz_spatial_data.loc[:, "image"] == image
        ]

        nn.fit(image_lz_spatial_data.loc[:, ["centroid-0", "centroid-1"]])
        dz_distances, _ = nn.kneighbors(
            image_dz_spatial_data.loc[:, ["centroid-0", "centroid-1"]], n_neighbors=k
        )

        nn.fit(image_dz_spatial_data.loc[:, ["centroid-0", "centroid-1"]])
        lz_distances, _ = nn.kneighbors(
            image_lz_spatial_data.loc[:, ["centroid-0", "centroid-1"]], n_neighbors=k
        )
        result.loc[image_lz_spatial_data.index, "distance_to_border"] = np.mean(
            lz_distances, axis=1
        )
        result.loc[image_dz_spatial_data.index, "distance_to_border"] = np.mean(
            dz_distances, axis=1
        )

        image_distances = result.loc[
            spatial_data.loc[spatial_data.loc[:, "image"] == image].index,
            "distance_to_border",
        ]
        result.loc[
            spatial_data.loc[spatial_data.loc[:, "image"] == image].index,
            "scaled_distance_to_border",
        ] = (image_distances - np.min(image_distances)) / (
            np.max(image_distances) - np.min(image_distances)
        )

        nn.fit(image_spatial_data.loc[:, ["centroid-0", "centroid-1"]])
        knns = nn.kneighbors(
            image_spatial_data.loc[:, ["centroid-0", "centroid-1"]],
            n_neighbors=k + 1,
            return_distance=False,
        )
        image_spatial_data_index = np.array(list(image_spatial_data.index))
        for i in range(len(knns)):
            idx = image_spatial_data_index[i]
            cell_types_knns = np.array(
                nuc_data.loc[image_spatial_data_index[knns[i][1:]], "cell_type"]
            )
            result.loc[idx, "frequency_of_dz_neighbors"] = np.mean(
                cell_types_knns == "DZ B-cells"
            )
            result.loc[idx, "frequency_based_distance_to_border"] = np.abs(
                2 * result.loc[idx, "frequency_of_dz_neighbors"] - 1
            )
    return result.dropna()
