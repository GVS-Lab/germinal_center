# Germinal Centers

The repository contains the code used to run the analysis of the chromatin states of the cell populations in germinal centers and selected regions of interest which is discussed in:

> [*TO BE ADDED*]()

----

# System requirements

The code has been developed and executed on a Thinkpad P1 mobile work station running Ubuntu 18.04 LTS with a Intel(R) Core(TM) i7-9750 CPU with 2.60 GHz, 32GB RAM and a Nvidia P200 GPU. Note that the code can also be run for machines without a GPU or with less available RAM.

## Installation

To install the code, please clone the repository and install the required software libraries and packages listed in the **requirements.txt** file:
```
git clone https://github.com/GVS-Lab/germinal_center.git
conda create --name gc --file requirements.txt
conda activate gc
```

## Data resouces (Optional)

Intermediate results of the analysis can be obtained from [*TO BE ADDED*]() but can also be produced using the steps described below to reproduce the results of the paper. If you want to use and/or adapt the code to run another analysis, the data is not required neither.

---

# Reproducing the paper results

## 1. Data preprocessing

The data preprocessing steps quantile-normalize the data, segment individual nuclei and cells as well as measure the chrometric features described in [Venkatachalapathy et al. (2020)](https://www.molbiolcell.org/doi/10.1091/mbc.E19-08-0420) for each nucleus and quantify the associated cellular expression of the proteins stained for in the processed immunofluorescent images. To preprocess the imaging data for the analysis of the B-cell populations in the germinal centers or the correlation analysis of the selected microimages please use the notebooks ```notebooks/dataset1/feature_generation.ipynb``` or ```notebooks/dataset3/feature_generation.ipynb``` respectively.

## 2. Analysis of the B-cell populations in germinal centers

## 3. Analysis of the gene expression and chromatin signature of cells

---

# How to cite

----
