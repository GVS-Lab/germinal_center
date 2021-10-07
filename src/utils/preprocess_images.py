# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from glob import glob
import imageio as imio
from tifffile import imsave
from pathlib import Path


def extract_channel_save_image(image_dir:str, output_dir:str, channel:int):
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # read in the images,segment and save labels
    all_images = sorted(glob(image_dir +'*.tif'))
        
    for i in range(len(all_images)):
        X = imio.imread(all_images[i]) # read
        X = X[:,:,-channel] #extract channel
        imsave(output_dir + all_images[i].rsplit('/', 1)[-1][:-4] +'.tif',  X) #save
