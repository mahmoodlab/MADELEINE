"""
# Usage
python extract_patch_embeddings.py  --local_dir ../results_brca/MADELEINE --slide_dir ../sample_data/bcnb/
"""

# general
import sys; sys.path.append("../")

import argparse
import os
import pickle

import geopandas as gpd
import openslide
from tqdm import tqdm

from core.preprocessing.conch_patch_embedder import embed_tiles
from core.preprocessing.hest_modules.segmentation import (get_tissue_vis,
                                                          segment_tissue_deep)
from core.preprocessing.hest_modules.wsi import get_pixel_size, OpenSlideWSI
from core.preprocessing.patching import extract_patch_coords

import pdb 

EXTENSIONS = ['.svs', '.mrxs', '.tiff', '.tif', '.TIFF', '.ndpi']

def segment(slide_dir, out_dir, patch_mag, patch_size, step_size):

    fnames = os.listdir(slide_dir)
    fnames = [fname for fname in fnames if any(fname.endswith(ext) for ext in EXTENSIONS)]
    print('* Running segmentation and patching on {} slides.'.format(len(fnames)))

    # make seg paths 
    seg_path = os.path.join(out_dir, 'segmentation')
    os.makedirs(os.path.join(seg_path, 'pkl'), exist_ok=True)  # QuPath compatibility. 
    os.makedirs(os.path.join(seg_path, 'geojson'), exist_ok=True)  # QuPath compatibility. 
    os.makedirs(os.path.join(seg_path, 'jpeg'), exist_ok=True)

    # make patch paths 
    patch_path = os.path.join(out_dir, 'patches')
    os.makedirs(patch_path, exist_ok=True)

    # make patch embedding paths
    patch_emb_path = os.path.join(out_dir, 'patch_embeddings')
    os.makedirs(patch_emb_path, exist_ok=True)

    for fn in tqdm(fnames):

        # segmentation 
        extension = '.' + fn.split('.')[-1]
        wsi = OpenSlideWSI(openslide.OpenSlide(os.path.join(slide_dir, fn)))

        # Paul: would be nice to decouple the pixel size from the WSI in order to support more formats
        pixel_size = get_pixel_size(wsi.img)
        gdf_contours = segment_tissue_deep(wsi, pixel_size, batch_size=64)

        # save thumbnail  
        seg_name = fn.replace(extension, '_tissue_mask.jpeg')
        get_tissue_vis(wsi, gdf_contours).save(os.path.join(seg_path, 'jpeg', seg_name))

        # save as geojson 
        seg_name = fn.replace(extension, '_tissue_mask.geojson')
        gdf_contours.to_file(os.path.join(seg_path, 'geojson', seg_name), driver="GeoJSON")

        # save as pickle 
        seg_name = fn.replace(extension, '_tissue_mask.pkl')
        with open(os.path.join(seg_path, 'pkl', seg_name), "wb") as f:
            pickle.dump(gdf_contours, f)

        # patching
        patch_name = fn.replace(extension, '_patches.h5')
        extract_patch_coords(
            wsi, 
            gdf_contours,
            save_path_hdf5=os.path.join(patch_path, patch_name),
            patch_mag=patch_mag,
            patch_size=patch_size,
            step_size=step_size
        )

        # extracting patch embeddings 
        patch_emb_name = fn.replace(extension, '_embeddings.h5')
        embed_tiles(
            wsi=wsi,
            tile_h5_path=os.path.join(patch_path, patch_name),
            embedding_save_path=os.path.join(patch_emb_path, patch_emb_name)
        )

    return None

def segment_old(slide_dir, out_dir, patch_mag, patch_size, step_size):

    fnames = os.listdir(slide_dir)
    fnames = [fname for fname in fnames if any(fname.endswith(ext) for ext in EXTENSIONS)]
    print('* Running segmentation and patching on {} slides.'.format(len(fnames)))

    # make seg paths 
    seg_path = os.path.join(out_dir, 'segmentation')
    os.makedirs(seg_path, exist_ok=True)
    os.makedirs(os.path.join(seg_path, 'pkl'), exist_ok=True)
    os.makedirs(os.path.join(seg_path, 'jpeg'), exist_ok=True)
    os.makedirs(os.path.join(seg_path, 'geojson'), exist_ok=True)  # QuPath compatibility. 

    # make patch paths 
    patch_path = os.path.join(out_dir, 'patches')
    os.makedirs(patch_path, exist_ok=True)

    # make patch embedding paths
    patch_emb_path = os.path.join(out_dir, 'patch_embeddings')
    os.makedirs(patch_emb_path, exist_ok=True)

    for fn in tqdm(fnames):

        # segmentation 
        extension = '.' + fn.split('.')[-1]
        wsi = openslide.OpenSlide(os.path.join(slide_dir, fn))
        pixel_size = get_pixel_size(wsi)
        gdf_contours = segment_tissue_deep(wsi, pixel_size, batch_size=64)

        # save thumbnail  
        seg_name = fn.replace(extension, '_tissue_mask.jpeg')
        get_tissue_vis(wsi, gdf_contours).save(os.path.join(seg_path, 'jpeg', seg_name))

        # save as geojson 
        seg_name = fn.replace(extension, '_tissue_mask.geojson')
        gdf_contours.to_file(os.path.join(seg_path, 'geojson', seg_name), driver="GeoJSON")

        # save as pickle 
        seg_name = fn.replace(extension, '_tissue_mask.pkl')
        with open(os.path.join(seg_path, 'pkl', seg_name), "wb") as f:
            pickle.dump(gdf_contours, f)

        # patching
        patch_name = fn.replace(extension, '_patches.h5')
        extract_patch_coords(
            wsi, 
            gdf_contours,
            save_path_hdf5=os.path.join(patch_path, patch_name),
            patch_mag=patch_mag,
            patch_size=patch_size,
            step_size=step_size
        )

        # extracting patch embeddings 
        patch_emb_name = fn.replace(extension, '_embeddings.h5')
        # embed_tiles(
        #     tile_h5_path=os.path.join(patch_path, patch_name),
        #     embedding_save_path=os.path.join(patch_emb_path, patch_emb_name)
        # )

    return None

if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_dir", type=str, help="Directory with slides.", default=None)
    parser.add_argument("--local_dir", type=str, help="Where to save tissue segmentation, patch coords, and patch embeddings.", default='./../results')
    parser.add_argument("--patch_mag", type=int, help="Magnification at which patching operates. Default to 10x.", default=10)
    parser.add_argument("--patch_size", type=int, help="Patch size. Default to 256.", default=256)
    parser.add_argument("--step_size", type=int, help="Step size when patching. Default to patch size.", default=None)

    args = parser.parse_args()

    if args.step_size is None: 
        args.step_size = args.patch_size

    print('*** Starting segmentation and patching ***')
    segment(args.slide_dir, args.local_dir, args.patch_mag, args.patch_size, args.step_size)
