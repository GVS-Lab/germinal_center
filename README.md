# Germinal Centers

The repository contains the code used to run the analysis of the chromatin states of the cell populations in germinal centers and selected regions of interest which is discussed in:

> [**TO BE ADDED**]()

<p align="center" width="100%">
  <b>Computational analysis pipeline</b> <br>
    <img width="66%" src="https://github.com/GVS-Lab/germinal_center/blob/bf3cfc13d338a21edf7db25b881ae6f60ea21a87/gc_overview.png">
</p>

----

# System requirements

The code has been developed and executed on a Thinkpad P1 mobile work station running Ubuntu 18.04 LTS with a Intel(R) Core(TM) i7-9750H CPU with 2.60 GHz, 32GB RAM and a Nvidia P200 GPU. Note that the code can also be run for machines without a GPU or with less available RAM.

## Installation

To install the code, please clone the repository and install the required software libraries and packages listed in the **requirements.txt** file:
```
git clone https://github.com/GVS-Lab/germinal_center.git
conda create --name gc --file requirements.txt
conda activate gc
```

## Data resouces (Optional)

Intermediate results of the analysis can be obtained from [https://drive.google.com/drive/folders/1HszNjSRFI2x25mEDQo-a_rKpemwtJZ4C?usp=sharing](here) but can also be produced using the steps described below to reproduce the results of the paper. If you want to use and/or adapt the code to run another analysis, the data is not required neither.

---

# Reproducing the paper results

## 1. Data preprocessing

The data preprocessing steps quantile-normalize the data, segment individual nuclei and cells as well as measure the chrometric features described in [Venkatachalapathy et al. (2020)](https://www.molbiolcell.org/doi/10.1091/mbc.E19-08-0420) for each nucleus and quantify the associated cellular expression of the proteins stained for in the processed immunofluorescent images. To preprocess the imaging data for the analysis of the B-cell populations in the germinal centers or the correlation analysis of the selected microimages please use the notebooks ```notebooks/dataset1/feature_generation.ipynb``` or ```notebooks/dataset3/feature_generation.ipynb``` respectively.

## 2. Analysis of the B-cell populations in germinal centers

To run the analysis regarding the different B-cell populations in the light respectively dark zone of the germinal centers, please use the code provided in the notebook ```notebooks/dataset1/light_vs_darkzone_bcells_and_tcell_integration.ipynb```.

## 3. Analysis of the gene expression and chromatin signature of cells

Finally, the correlation analysis of the measured dark zone gene expression signatures of the selected RoIs and the chromatin states corresponding to those can be run using the code in ```notebooks/dataset3/chrometric_correlation_analysis.ipynb```.

---

# How to cite

If you use any of the code or resources provided here please make sure to reference the required software libraries if needed and also cite our work:

**TO BE ADDED**

----
