import os 
import time 
import math 
import cv2
import numpy as np
import multiprocessing as mp

from core.preprocessing.conch_patch_embedder import save_hdf5
from core.preprocessing.hest_modules.wsi import get_pixel_size, WSI


def magnification_to_pixel_size(mag, ref_mag=40, ref_px_size=0.25):
    """
    Calculates the pixel size in microns per pixel for a given magnification level.

    Parameters:
        mag (float or int): The desired magnification level (e.g., 40, 20, 10).
        ref_mag (float or int, optional): The reference magnification level 
            for which the reference pixel size is known. Defaults to 40x.
        ref_px_size (float, optional): The pixel size in microns per pixel at the 
            reference magnification level. Defaults to 0.25 microns per pixel for 40x magnification.

    Returns:
        float: Pixel size in microns per pixel for the given magnification level.
    """
    if mag <= 0:
        raise ValueError("Magnification level must be greater than zero.")

    # Calculate the pixel size based on the magnification
    pixel_size = ref_px_size * (ref_mag / mag)
    return pixel_size


def extract_patch_coords(wsi, contours_tissue, save_path_hdf5, patch_mag, patch_size=256, step_size=0):
    """
    Patching WSI based on tissue contours and saves them to HDF5.

    Parameters:
        wsi (openslide.OpenSlide): OpenSlide.
        
        contours_tissue (gpd.GeoDataFrame): Contours.
        
        save_path_hdf5 (str): The file path where the extracted patch coordinates will be saved in HDF5 format.
        
        patch_mag (int): Target magnification at which patches should be extracted from the WSI, e.g., 10x, 20x.
        
        patch_size (int, optional): Target size of the patches to extract, specified in pixels. 
            Defaults to 256, meaning each patch will be 256x256 pixels.
        
        step_size (int, optional): Target step size in pixels between adjacent patches. 
            Defaults to 256, meaning patches are extracted with no overlap.
    """
    import pandas as pd
    import geopandas as gpd

    src_pixel_size = get_pixel_size(wsi.img)
    dst_pixel_size = magnification_to_pixel_size(patch_mag)

    n_contours = len(contours_tissue)
    print("Total number of contours to process: ", n_contours)
    fp_chunk_size = math.ceil(n_contours * 0.05)
    init = True
    for idx, row in contours_tissue.iterrows():
        if (idx + 1) % fp_chunk_size == fp_chunk_size:
            print('Processing contour {}/{}'.format(idx, n_contours))
        
        cont = gpd.GeoDataFrame(pd.DataFrame(row)[1:].transpose())
        overlap = int(np.clip(patch_size - step_size, 0, None))
        asset_dict, attr_dict = process_contour(wsi, cont, src_pixel_size, dst_pixel_size, patch_size, overlap)
        if len(asset_dict) > 0:
            if init:
                save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')
                init = False
            else:
                save_hdf5(save_path_hdf5, asset_dict, mode='a')
    return None 


def process_contour(wsi: WSI, cont, src_pixel_size, dst_pixel_size, patch_size = 256, overlap = 0, name='default'):

    patcher = wsi.create_patcher(patch_size, src_pixel_size, dst_pixel_size, overlap=overlap, mask=cont, coords_only=True)
    results = np.array([[int(coords[0]), int(coords[1])] for coords in patcher])
    patch_level = patcher.level
    level_downsample = wsi.level_downsamples()[patcher.level]
    level_dimensions = wsi.level_dimensions()[patcher.level]

    # extra downsample applied on top of level downsample
    custom_downsample = patcher.downsample / level_downsample
    
    print('Extracted {} coordinates'.format(len(results)))

    if len(results)>0:
        asset_dict = {'coords': results}
        
        # Why aren't we giving the real dowsample here?
        attr = {
            'patch_size' :            patch_size, # To be considered...
            'patch_level' :           patch_level,
            'downsample':             level_downsample,
            'custom_downsample':      custom_downsample,
            'downsampled_level_dim': level_dimensions,
            'level_dim':              level_downsample,
            'name':                   name,
        }

        attr_dict = { 'coords' : attr}
        return asset_dict, attr_dict

    else:
        return {}, {}

