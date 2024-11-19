import os

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from conch.open_clip_custom import create_model_from_pretrained
from hestcore.datasets import WSIPatcherDataset
# from core.preprocessing.hest_modules.wsi import WSIPatcher
from hestcore.wsi import OpenSlideWSIPatcher
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from core.utils.utils import get_pixel_size, mag_to_px_size


def save_hdf5(output_fpath, 
                  asset_dict, 
                  attr_dict= None, 
                  mode='a', 
                  auto_chunk = True,
                  chunk_size = None):
    """
    output_fpath: str, path to save h5 file
    asset_dict: dict, dictionary of key, val to save
    attr_dict: dict, dictionary of key: {k,v} to save as attributes for each key
    mode: str, mode to open h5 file
    auto_chunk: bool, whether to use auto chunking
    chunk_size: if auto_chunk is False, specify chunk size
    """
    with h5py.File(output_fpath, mode) as f:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if len(data_shape) == 1:
                val = np.expand_dims(val, axis=1)
                data_shape = val.shape

            if key not in f: # if key does not exist, create dataset
                data_type = val.dtype
                if data_type == np.object_: 
                    data_type = h5py.string_dtype(encoding='utf-8')
                if auto_chunk:
                    chunks = True # let h5py decide chunk size
                else:
                    chunks = (chunk_size,) + data_shape[1:]
                try:
                    dset = f.create_dataset(key, 
                                            shape=data_shape, 
                                            chunks=chunks,
                                            maxshape=(None,) + data_shape[1:],
                                            dtype=data_type)
                    ### Save attribute dictionary
                    if attr_dict is not None:
                        if key in attr_dict.keys():
                            for attr_key, attr_val in attr_dict[key].items():
                                dset.attrs[attr_key] = attr_val
                    dset[:] = val
                except:
                    print(f"Error encoding {key} of dtype {data_type} into hdf5")
                
            else:
                dset = f[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                assert dset.dtype == val.dtype
                dset[-data_shape[0]:] = val
    
    return output_fpath


def collate_features(batch):
    features = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return features, coords


class ConchTileEmbedder:
    def __init__(self, 
                 model_name='conch_ViT-B-16',
                 model_repo='hf_hub:MahmoodLab/conch',
                 target_patch_size=256,
                 target_mag=20,
                 device='cuda',
                 precision=torch.float32,
                 save_path=None):
        self.model_name = model_name
        self.model_repo = model_repo
        self.device = device
        self.precision = precision
        self.save_path = save_path
        self.target_patch_size = target_patch_size
        self.target_mag = target_mag
        self.model, self.img_transforms = self._build_conch_model()

    def _build_conch_model(self):
        model, eval_transform = create_model_from_pretrained(self.model_name, self.model_repo, force_image_size=224)
        return model, eval_transform

    def embed_tiles(self, wsi, gdf_contours, fn) -> str:

        # set i/o paths
        patching_save_path = os.path.join(self.save_path, 'patches', f'{fn}_patches.png')
        embedding_save_path = os.path.join(self.save_path, 'patch_embeddings', f'{fn}.h5')

        dst_pixel_size = mag_to_px_size(self.target_mag)
        src_pixel_size = get_pixel_size(wsi.img)

        patcher = wsi.create_patcher(
            self.target_patch_size,
            src_pixel_size,
            dst_pixel_size,
            mask=gdf_contours,
            pil=True
        )

        conch_transforms = transforms.Compose([
            self.img_transforms, 
            transforms.Lambda(lambda x: torch.unsqueeze(x, 0))
        ])

        dataset = WSIPatcherDataset(patcher, transform=conch_transforms)
        
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=64, 
            shuffle=False,
            num_workers=8,
            collate_fn=collate_features,
        )

        self.model.to(self.device)
        self.model.eval()
        
        for batch_idx, (imgs, coords) in tqdm(enumerate(dataloader), total=len(dataloader)):
            imgs = imgs.to(self.device)
            with torch.inference_mode(), torch.amp.autocast(dtype=self.precision, device_type=self.device):
                embeddings = self.model.encode_image(imgs, proj_contrast=False, normalize=False)
            mode = 'w' if batch_idx == 0 else 'a'
            asset_dict = {
                'features': embeddings.cpu().numpy(),
                'coords': coords,
            }
            save_hdf5(embedding_save_path, mode=mode, asset_dict=asset_dict)
        
        return embedding_save_path