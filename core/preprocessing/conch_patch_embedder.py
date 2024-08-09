from tqdm import tqdm 
import numpy as np 
import h5py 
import os
from PIL import Image

import torch 
from torch.utils.data import Dataset

from conch.open_clip_custom import create_model_from_pretrained

# from core.preprocessing.hest_modules.wsi import WSIPatcher
from core.preprocessing.hest_modules.wsi import OpenSlideWSIPatcher, get_pixel_size


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


class TileEmbedder:
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

        dataset = TileDataset(
            wsi=wsi,
            gdf_contours=gdf_contours,
            target_patch_size=self.target_patch_size,
            target_mag=self.target_mag,
            eval_transform=self.img_transforms,
            save_path=patching_save_path)
        
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


class TileDataset(Dataset):
    def __init__(self, wsi, gdf_contours, target_patch_size, target_mag, eval_transform, save_path=None):
        self.wsi = wsi
        self.gdf_contours = gdf_contours
        self.eval_transform = eval_transform

        self.patcher = OpenSlideWSIPatcher(
            wsi=wsi,
            patch_size=target_patch_size,
            src_pixel_size=get_pixel_size(wsi.img),
            dst_pixel_size=self.mag_to_px_size(target_mag),
            mask=gdf_contours,
            coords_only=False,
        )
        self.patcher.save_visualization(path=save_path)

    @staticmethod
    def mag_to_px_size(mag):
        if mag == 5: return 2.0
        if mag == 10: return 1.0
        if mag == 20: return 0.5
        if mag == 40: return 0.25
        else: raise ValueError('Magnification should be in [5, 10, 20, 40].')

    # def _load_coords(self):
    # 	with h5py.File(self.coords_h5_fpath, "r") as f:
    # 		self.attr_dict = {k: dict(f[k].attrs) for k in f.keys() if len(f[k].attrs) > 0}
    # 		self.coords = f['coords'][:]
    # 		self.patch_size = f['coords'].attrs['patch_size']
    # 		self.custom_downsample = f['coords'].attrs['custom_downsample']
    # 		self.target_patch_size = int(self.patch_size) // int(self.custom_downsample) if self.custom_downsample > 1 else self.patch_size

    def __len__(self):
        return len(self.patcher)

    def __getitem__(self, idx):
        img, x, y = self.patcher[idx]
        img = Image.fromarray(img, 'RGB')
        img = self.eval_transform(img).unsqueeze(dim=0)
        return img, (x, y)
