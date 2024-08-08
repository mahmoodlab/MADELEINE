import sys; sys.path.append('../')
import argparse
import os
import pickle
import logging

import openslide
from tqdm import tqdm

from core.preprocessing.conch_patch_embedder import embed_tiles
from core.preprocessing.hest_modules.segmentation import segment_tissue_deep
from core.preprocessing.hest_modules.wsi import get_pixel_size, OpenSlideWSI
from core.preprocessing.patching import extract_patch_coords

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File extensions for slide images
EXTENSIONS = ['.svs', '.mrxs', '.tiff', '.tif', '.TIFF', '.ndpi']

def segment(slide_dir, out_dir, patch_mag, patch_size, step_size):
    fnames = [fname for fname in os.listdir(slide_dir) if any(fname.endswith(ext) for ext in EXTENSIONS)]
    logger.info(f'Running segmentation, patching, and feature extraction on {len(fnames)} slides.')

    # Create necessary directories
    seg_path = os.path.join(out_dir, 'segmentation')
    os.makedirs(seg_path, exist_ok=True)

    patch_path = os.path.join(out_dir, 'patches')
    os.makedirs(patch_path, exist_ok=True)

    patch_emb_path = os.path.join(out_dir, 'patch_embeddings')
    os.makedirs(patch_emb_path, exist_ok=True)

    for fn in tqdm(fnames):
        wsi = OpenSlideWSI(openslide.OpenSlide(os.path.join(slide_dir, fn)))
        pixel_size = get_pixel_size(wsi.img)
        fn_no_extension = os.path.splitext(fn)[0]

        gdf_contours = segment_tissue_deep(
            wsi=wsi,
            pixel_size=pixel_size,
            batch_size=64,
            save_path=seg_path,
            save_bn=fn_no_extension,
        )

        patch_name = f'{fn_no_extension}_patches.h5'
        extract_patch_coords(
            wsi, 
            gdf_contours,
            save_path_hdf5=os.path.join(patch_path, patch_name),
            patch_mag=patch_mag,
            patch_size=patch_size,
            step_size=step_size
        )

        patch_emb_name = f'{fn_no_extension}_embeddings.h5'
        embed_tiles(
            wsi=wsi,
            tile_h5_path=os.path.join(patch_path, patch_name),
            embedding_save_path=os.path.join(patch_emb_path, patch_emb_name)
        )

    logger.info('Done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_dir", type=str, help="Directory with slides.", default=None)
    parser.add_argument("--local_dir", type=str, help="Where to save tissue segmentation, patch coords, and patch embeddings.", default='./../results')
    parser.add_argument("--patch_mag", type=int, help="Magnification at which patching operates. Default to 10x.", default=10)
    parser.add_argument("--patch_size", type=int, help="Patch size. Default to 256.", default=256)
    parser.add_argument("--step_size", type=int, help="Step size when patching. Default to patch size.", default=None)

    args = parser.parse_args()

    if args.step_size is None: 
        args.step_size = args.patch_size

    logger.info('Initiate run...')
    segment(args.slide_dir, args.local_dir, args.patch_mag, args.patch_size, args.step_size)
