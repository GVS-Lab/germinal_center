# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.manifold import TSNE
from sklearn.metrics import plot_roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import seaborn as sns
from tqdm import tqdm



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



def plot_cv_conf_mtx(X, y, model, cv=10, figsize=[6, 4]):
    avg_conf_mtx = compute_avg_conf_mtx(model=model, n_folds=cv, features=X, labels=y)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(avg_conf_mtx, annot=True, cmap="BuPu", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.title("Average confusion matrix from {}-fold stratified CV".format(cv))
    plt.show()
    plt.close()

    

def compute_avg_conf_mtx(model, n_folds, features, labels):
    skf = StratifiedKFold(n_folds)
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
    confusion_mtx = confusion_mtx / n_folds
    return pd.DataFrame(
        confusion_mtx, index=sorted(set(labels)), columns=sorted(set(labels))
    )


    
def plot_feature_importance(importance, names, model_type, n_features=20):
    
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
    plt.figure(figsize=(8, 6))
    
    # Plot Searborn bar chart
    sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"], color= 'dimgray')
    
    # Add chart labels
    plt.title(model_type + "Feature importance")
    plt.xlabel("Feature importance")
    plt.ylabel("Featues")
    plt.show()


def find_markers(data, labels):
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

            pval = stats.ttest_ind(x, y, equal_var=False)[1]
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
    result["fwer_padj"] = result.pval * i
    result.loc[result["fwer_padj"] > 1, "fwer_padj"] = 1
    return result.sort_values("fwer_padj")
