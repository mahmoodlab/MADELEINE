import sys; sys.path.append('../')
import argparse
import logging
import os

import openslide
from tqdm import tqdm
from core.utils.utils import get_pixel_size

from core.preprocessing.conch_patch_embedder import ConchTileEmbedder
from hestcore.wsi import OpenSlideWSI
from hestcore.segmentation import segment_tissue_deep

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
    embedder = ConchTileEmbedder(target_patch_size=patch_size, target_mag=patch_mag, save_path=out_dir)

    for fn in tqdm(fnames):

        # 1. read slide 
        wsi = OpenSlideWSI(openslide.OpenSlide(os.path.join(slide_dir, fn)))
        pixel_size = get_pixel_size(wsi.img)
        fn_no_extension = os.path.splitext(fn)[0]

        # 2. segment tissue 
        gdf_contours = segment_tissue_deep(
            wsi,
            pixel_size,
            batch_size=64
        )

        # 3. save segmentation + visualization
        os.makedirs(os.path.join(out_dir, 'geojson'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'jpeg'), exist_ok=True)
        seg_name = fn_no_extension + '_tissue_vis.jpeg'
        wsi.get_tissue_vis(gdf_contours).save(os.path.join(out_dir, 'jpeg', seg_name))
        seg_name = fn_no_extension + '_tissue_mask.geojson'
        gdf_contours.to_file(os.path.join(out_dir, 'geojson', seg_name), driver="GeoJSON")

        # 4. extract patches and embeddings
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
