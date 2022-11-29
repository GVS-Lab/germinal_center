# -*- coding: utf-8 -*-

import numpy as np


def clean_data(data, drop_columns=[], index_col=None):
    filtered_data = data.copy()

    if index_col is not None:
        filtered_data.index = data.loc[:, index_col]

    filtered_data = filtered_data.select_dtypes(include=np.number)
    data = filtered_data.loc[:, (filtered_data != filtered_data.iloc[0]).any()]
    data = data.dropna(axis=1)

    print(
        "Removed {} constant or features with missing values. Remaining: {}.".format(
            len(filtered_data.columns) - len(data.columns), len(data.columns)
        )
    )

    cleaned_data = data.drop(
        columns=list(set(drop_columns).intersection(set(data.columns)))
    )
    print(
        "Removed additional {} features. Remaining: {}.".format(
            len(data.columns) - len(cleaned_data.columns), len(cleaned_data.columns)
        )
    )
    n_samples = len(cleaned_data)
    cleaned_data = cleaned_data.dropna(axis=0)
    print(
        "Removed {} samples with missing values. Remaining: {}.".format(
            n_samples - len(cleaned_data), len(cleaned_data)
        )
    )
    return cleaned_data


def remove_correlated_features(data, threshold):
    data_corr = data.corr().abs()
    upper = data_corr.where(np.triu(np.ones(data_corr.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(
        "Removed {}/{} features with a Pearson correlation above {}. Remaining: {}".format(
            len(to_drop),
            len(data.columns),
            threshold,
            len(data.columns) - len(to_drop),
        )
    )
    return data.drop(to_drop, axis=1)
