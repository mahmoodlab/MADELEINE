import sys; sys.path.append('../')
import argparse
import os
import logging

import openslide
from tqdm import tqdm

from core.preprocessing.conch_patch_embedder import TileEmbedder
from core.preprocessing.hest_modules.segmentation import TissueSegmenter
from core.preprocessing.hest_modules.wsi import get_pixel_size, OpenSlideWSI

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# File extensions for slide images
EXTENSIONS = ['.svs', '.mrxs', '.tiff', '.tif', '.TIFF', '.ndpi']

def process(slide_dir, out_dir, patch_mag, patch_size):
    fnames = [fname for fname in os.listdir(slide_dir) if any(fname.endswith(ext) for ext in EXTENSIONS)]
    logger.info(f'Running segmentation, patching, and feature extraction on {len(fnames)} slides.')

    # Create necessary directories
    out_dir = os.path.join(out_dir, 'processing_conch_nWSI_{}_mag_{}x_patchsize_{}'.format(
        len(fnames),
        patch_mag,
        patch_size
    ))
    seg_path = os.path.join(out_dir, 'segmentation')
    os.makedirs(seg_path, exist_ok=True)

    patch_path = os.path.join(out_dir, 'patches')
    os.makedirs(patch_path, exist_ok=True)

    patch_emb_path = os.path.join(out_dir, 'patch_embeddings')
    os.makedirs(patch_emb_path, exist_ok=True)

    # create tissue segmenter and tile embedder
    segmenter = TissueSegmenter(save_path=seg_path, batch_size=64)
    embedder = TileEmbedder(target_patch_size=patch_size, target_mag=patch_mag, save_path=out_dir)

    for fn in tqdm(fnames):

        # 1. read slide 
        wsi = OpenSlideWSI(openslide.OpenSlide(os.path.join(slide_dir, fn)))
        pixel_size = get_pixel_size(wsi.img)
        fn_no_extension = os.path.splitext(fn)[0]

        # 2. segment tissue 
        gdf_contours = segmenter.segment_tissue(
            wsi=wsi,
            pixel_size=pixel_size,
            save_bn=fn_no_extension,
        )

        # 3. extract patches and embeddings
        embedder.embed_tiles(
            wsi=wsi,
            gdf_contours=gdf_contours,
            fn=fn_no_extension,
        )

    logger.info('Done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_dir", type=str, help="Directory with slides.", default=None)
    parser.add_argument("--local_dir", type=str, help="Where to save tissue segmentation, patch coords, and patch embeddings.", default='./../data/downstream')
    parser.add_argument("--patch_mag", type=int, help="Magnification at which patching operates. Default to 10x.", default=10)
    parser.add_argument("--patch_size", type=int, help="Patch size. Default to 256.", default=256)

    args = parser.parse_args()

    logger.info('Initiate run...')
    process(args.slide_dir, args.local_dir, args.patch_mag, args.patch_size)
