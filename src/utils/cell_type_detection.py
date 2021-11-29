# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn import mixture
import scipy.spatial as ss


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


def t_cell_neighbours(dataset, R=1):
    
    image_ids = dataset['image'].unique()
    grouped = dataset.groupby(dataset.image)

    tcell_neighbours = []
    
    for i in range(len(image_ids)):
        data = grouped.get_group(image_ids[i])
        
        t_cells = data['nuc_id'][data['type'] == 't_cells']
        cords=np.column_stack((data['centroid-0'],data['centroid-1']))
        #obtain the distance matrix 
        dist_matrix=ss.distance.squareform(ss.distance.pdist(cords, 'euclidean'))
        dist_matrix = pd.DataFrame(dist_matrix)
        dist_matrix.columns = data['nuc_id']

        #Defining neighbourhood radius "R" and counting the number of nuclei in "R"
        t_cell_neighbours = dist_matrix[dist_matrix.columns.intersection(t_cells)]
        mask=((t_cell_neighbours<R) & (t_cell_neighbours >0)).astype(float)
        status = np.nansum(mask, axis=1)

        tcell_neighbours.extend(data['nuc_id'][status>=1])
    
    return tcell_neighbours