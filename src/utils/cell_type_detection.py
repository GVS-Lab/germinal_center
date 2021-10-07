# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn import mixture


def get_postive_cells_batch(dataset, img_names):

    postive_ids = []
    
    for i in range(len(img_names)):
        img_subset = dataset[dataset["image"] == img_names[i]]
        postive_ids.extend(get_postive_cells(img_subset))

    
    return postive_ids



def get_postive_cells(dataset, feature  = 'int_mean', id_to_return = 'nuc_id'):

    dat = np.array(dataset[feature]).reshape(-1, 1)
    
    positive_cells = dataset[assign_cell_status(dat)][id_to_return].tolist()
    
    return positive_cells
    
def assign_cell_status(dat):
    
    thresh = get_two_compoent_threshold(dat)
        
    status = dat > thresh
    
    return status
    

def get_two_compoent_threshold(data):

    gm = mixture.GaussianMixture(n_components=2, random_state=0).fit(data)
    threshold = np.mean(gm.means_)
    
    return threshold