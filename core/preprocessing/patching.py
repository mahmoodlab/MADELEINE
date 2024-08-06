import os 
import time 
import math 
import cv2
import numpy as np
import multiprocessing as mp

from core.preprocessing.conch_patch_embedder import save_hdf5
from core.preprocessing.wsi import get_pixel_size


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


def extract_patch_coords(wsi, contours_tissue, save_path_hdf5, patch_mag, patch_size=256, step_size=256):
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
    px_size = get_pixel_size(wsi)
    target_px_size = magnification_to_pixel_size(patch_mag)
    target_level = wsi.get_best_level_for_downsample(target_px_size / px_size)

    n_contours = len(contours_tissue)
    print("Total number of contours to process: ", n_contours)
    fp_chunk_size = math.ceil(n_contours * 0.05)
    init = True
    for idx, cont in enumerate(contours_tissue):
        if (idx + 1) % fp_chunk_size == fp_chunk_size:
            print('Processing contour {}/{}'.format(idx, n_contours))
        
        asset_dict, attr_dict = process_contour(wsi, cont, target_level, patch_size, step_size)
        if len(asset_dict) > 0:
            if init:
                save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')
                init = False
            else:
                save_hdf5(save_path_hdf5, asset_dict, mode='a')
    return None 


def process_contour(
        wsi, cont, contour_holes, patch_level, patch_size = 256, step_size = 256, use_padding=True, top_left=None, bot_right=None):

    start_x, start_y, w, h = cv2.boundingRect(cont)

    # @TODO: to be adapted 
    patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
    ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])
    
    img_w, img_h = self.level_dim[0]
    if use_padding:
        stop_y = start_y+h
        stop_x = start_x+w
    else:
        stop_y = min(start_y+h, img_h-ref_patch_size[1]+1)
        stop_x = min(start_x+w, img_w-ref_patch_size[0]+1)
    
    print("Bounding Box:", start_x, start_y, w, h)
    print("Contour Area:", cv2.contourArea(cont))

    if bot_right is not None:
        stop_y = min(bot_right[1], stop_y)
        stop_x = min(bot_right[0], stop_x)
    if top_left is not None:
        start_y = max(top_left[1], start_y)
        start_x = max(top_left[0], start_x)

    if bot_right is not None or top_left is not None:
        w, h = stop_x - start_x, stop_y - start_y
        if w <= 0 or h <= 0:
            print("Contour is not in specified ROI, skip")
            return {}, {}
        else:
            print("Adjusted Bounding Box:", start_x, start_y, w, h)

    cont_check_fn = IsInContour(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)

    step_size_x = step_size * patch_downsample[0]
    step_size_y = step_size * patch_downsample[1]

    x_range = np.arange(start_x, stop_x, step=step_size_x)
    y_range = np.arange(start_y, stop_y, step=step_size_y)
    x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
    coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

    num_workers = mp.cpu_count()
    if num_workers > 4:
        num_workers = 4
    pool = mp.Pool(num_workers)

    iterable = [(coord, contour_holes, ref_patch_size[0], cont_check_fn) for coord in coord_candidates]
    results = pool.starmap(WholeSlideImage.process_coord_candidate, iterable)
    pool.close()
    results = np.array([result for result in results if result is not None])
    
    print('Extracted {} coordinates'.format(len(results)))

    if len(results)>0:
        asset_dict = {'coords' :          results}
        
        attr = {
            'patch_size' :            patch_size, # To be considered...
            'patch_level' :           patch_level,
            'downsample':             self.level_downsamples[patch_level],
            'downsampled_level_dim' : tuple(np.array(self.level_dim[patch_level])),
            'level_dim':              self.level_dim[patch_level],
            'name':                   self.name,
        }

        attr_dict = { 'coords' : attr}
        return asset_dict, attr_dict

    else:
        return {}, {}


# Easy version of 4pt contour checking function - 1 of 4 points need to be in the contour for test to pass
class IsInContour(object):
     
	def __init__(self, contour, patch_size, center_shift=0.5):
		self.cont = contour
		self.patch_size = patch_size
		self.shift = int(patch_size//2*center_shift)
          
	def __call__(self, pt): 
		center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
		if self.shift > 0:
			all_points = [(center[0]-self.shift, center[1]-self.shift),
						  (center[0]+self.shift, center[1]+self.shift),
						  (center[0]+self.shift, center[1]-self.shift),
						  (center[0]-self.shift, center[1]+self.shift)
						  ]
		else:
			all_points = [center]
		
		for points in all_points:
			if cv2.pointPolygonTest(self.cont, tuple(np.array(points).astype(float)), False) >= 0:
				return 1
		return 0
